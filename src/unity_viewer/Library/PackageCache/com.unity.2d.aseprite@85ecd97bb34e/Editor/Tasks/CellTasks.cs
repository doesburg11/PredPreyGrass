using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;

namespace UnityEditor.U2D.Aseprite
{
    internal static class CellTasks
    {
        public static void GetCellsFromLayers(List<Layer> layers, out List<Cell> cells)
        {
            cells = new List<Cell>();
            for (var i = 0; i < layers.Count; ++i)
                cells.AddRange(layers[i].cells);
        }

        public static Dictionary<int, List<Cell>> GetAllCellsPerFrame(IReadOnlyList<Layer> layers)
        {
            var cellsToSortingIndex = new Dictionary<Cell, int>();
            var cellsPerFrame = new Dictionary<int, List<Cell>>();
            for (var i = 0; i < layers.Count; ++i)
            {
                var layer = layers[i];
                var cells = layer.cells;
                for (var m = 0; m < cells.Count; ++m)
                {
                    var cell = cells[m];
                    var width = cell.cellRect.width;
                    var height = cell.cellRect.height;
                    if (width == 0 || height == 0)
                        continue;

                    if (cellsPerFrame.TryGetValue(cell.frameIndex, out var frame))
                        frame.Add(cell);
                    else
                        cellsPerFrame.Add(cell.frameIndex, new List<Cell>() { cell });
                    
                    cellsToSortingIndex.Add(cell, layer.index + cell.additiveSortOrder);
                }

                var linkedCells = layer.linkedCells;
                for (var m = 0; m < linkedCells.Count; ++m)
                {
                    var frameIndex = linkedCells[m].frameIndex;
                    var linkedToFrame = linkedCells[m].linkedToFrame;

                    var cellIndex = cells.FindIndex(x => x.frameIndex == linkedToFrame);
                    Assert.AreNotEqual(-1, cellIndex, $"Linked Cell: {frameIndex} is linked to cell: {linkedToFrame}, which cannot be found.");

                    var cell = cells[cellIndex];

                    var width = cell.cellRect.width;
                    var height = cell.cellRect.height;
                    if (width == 0 || height == 0)
                        continue;

                    if (cellsPerFrame.TryGetValue(frameIndex, out var frame))
                        frame.Add(cell);
                    else
                        cellsPerFrame.Add(frameIndex, new List<Cell>() { cell });
                    
                    // Since the cell is linked, it may have already been added. If so, use the previous sorting index.
                    cellsToSortingIndex.TryAdd(cell, layer.index + cell.additiveSortOrder);
                }
            }
            
            // Sort the cell lists in the order of their cell index.
            // The cell index is a combination of the layer index and the Z-Index of the cell.
            foreach (var cellList in cellsPerFrame.Values)
            {
                cellList.Sort((x, y) =>
                {
                    var indexX = cellsToSortingIndex[x];
                    var indexY = cellsToSortingIndex[y];
                    if (indexX > indexY)
                        return 1;
                    if (indexX < indexY)
                        return -1;
                    return x.additiveSortOrder.CompareTo(y.additiveSortOrder);
                });
            }

            return cellsPerFrame;
        }
        
        public static List<Cell> MergeCells(IReadOnlyDictionary<int, List<Cell>> cellsPerFrame, string cellName)
        {
            var mergedCells = new List<Cell>(cellsPerFrame.Count);
            foreach (var (frameIndex, cells) in cellsPerFrame)
            {
                unsafe
                {
                    var count = cells.Count;

                    var textures = new NativeArray<IntPtr>(count, Allocator.Persistent);
                    var cellRects = new NativeArray<RectInt>(count, Allocator.Persistent);
                    var cellBlendModes = new NativeArray<BlendModes>(count, Allocator.Persistent);

                    for (var i = 0; i < cells.Count; ++i)
                    {
                        textures[i] = (IntPtr)cells[i].image.GetUnsafePtr();
                        cellRects[i] = cells[i].cellRect;
                        cellBlendModes[i] = cells[i].blendMode;
                    }

                    TextureTasks.MergeTextures(in textures, in cellRects, in cellBlendModes, out var output);
                    var mergedCell = new Cell()
                    {
                        cellRect = output.rect,
                        image = output.image,
                        frameIndex = frameIndex,
                        name = ImportUtilities.GetCellName(cellName, frameIndex, cellsPerFrame.Count, true),
                        spriteId = GUID.Generate()
                    };
                    mergedCells.Add(mergedCell);

                    textures.Dispose();
                    cellRects.Dispose();
                    cellBlendModes.Dispose();
                }
            }

            return mergedCells;
        }

        public static void CollectDataFromCells(List<Cell> cells, out List<NativeArray<Color32>> cellBuffers, out List<int2> cellSize)
        {
            cellBuffers = new List<NativeArray<Color32>>();
            cellSize = new List<int2>();

            for (var m = 0; m < cells.Count; ++m)
            {
                var size = cells[m].cellRect.size;
                if (size.x == 0 || size.y == 0)
                    continue;

                cellBuffers.Add(cells[m].image);
                cellSize.Add(new int2(size.x, size.y));
            }
        }

        public static void FlipCellBuffers(List<NativeArray<Color32>> imageBuffers, List<int2> cellSize)
        {
            unsafe
            {
                for (var i = 0; i < imageBuffers.Count; ++i)
                {
                    var buffer = imageBuffers[i];
                    TextureTasks.FlipTextureY((Color32*)buffer.GetUnsafePtr(), buffer.Length, cellSize[i]);
                    imageBuffers[i] = buffer;
                }   
            }
        }
    }
}
