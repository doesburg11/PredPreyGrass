using System.Collections.Generic;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace UnityEditor.U2D.Aseprite
{
    internal static class ImportMergedLayers
    {
        public static void Import(string assetName, List<Layer> layers, out List<NativeArray<Color32>> cellBuffers, out List<int2> cellSize)
        {
            var cellsPerFrame = CellTasks.GetAllCellsPerFrame(layers);
            var mergedCells = CellTasks.MergeCells(cellsPerFrame, assetName);

            CellTasks.CollectDataFromCells(mergedCells, out cellBuffers, out cellSize);
            UpdateLayerList(mergedCells, assetName, layers);
        }

        static void UpdateLayerList(List<Cell> cells, string assetName, List<Layer> layers)
        {
            layers.Clear();
            var flattenLayer = new Layer()
            {
                layerType = LayerTypes.Normal,
                cells = cells,
                index = 0,
                name = assetName
            };
            var guid = (uint)Layer.GenerateGuid(flattenLayer, layers);
            flattenLayer.uuid = new UUID(guid, 0, 0, 0);
            
            layers.Add(flattenLayer);
        }
    }
}
