using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Burst;

namespace UnityEditor.U2D.PSD
{
    internal static class ExtractLayerTask
    {
        struct LayerGroupData
        {
            public int startIndex { get; set; }
            public int endIndex { get; set; }
            
            /// <summary>
            /// The layer's bounding box in document space.
            /// </summary>
            public int4 documentRect { get; set; }
        }
        
        [BurstCompile]
        struct ConvertBufferJob : IJobParallelFor
        {
            [ReadOnly, DeallocateOnJobCompletion] 
            public NativeArray<int> inputTextureBufferSizes;
            [ReadOnly, DeallocateOnJobCompletion]
            public NativeArray<IntPtr> inputTextures;
            [ReadOnly, DeallocateOnJobCompletion]
            public NativeArray<int4> inputLayerRects;
            [ReadOnly, DeallocateOnJobCompletion]
            public NativeArray<LayerGroupData> layerGroupDataData;
            
            [ReadOnly, DeallocateOnJobCompletion] 
            public NativeArray<int4> outputLayerRect;
            [DeallocateOnJobCompletion]
            public NativeArray<IntPtr> outputTextures;
            public unsafe void Execute(int groupIndex)
            {
                var outputColor = (Color32*)outputTextures[groupIndex];
                var groupStartIndex = layerGroupDataData[groupIndex].startIndex;
                var groupEndIndex = layerGroupDataData[groupIndex].endIndex;

                var outStartX = outputLayerRect[groupIndex].x;
                var outStartY = outputLayerRect[groupIndex].y;
                var outWidth = outputLayerRect[groupIndex].z;
                var outHeight = outputLayerRect[groupIndex].w;

                for (var layerIndex = groupEndIndex; layerIndex >= groupStartIndex; --layerIndex)
                {
                    if (inputTextures[layerIndex] == IntPtr.Zero)
                        continue;

                    var inputColor = (Color32*)inputTextures[layerIndex];
                    var inX = inputLayerRects[layerIndex].x;
                    var inY = inputLayerRects[layerIndex].y;
                    var inWidth = inputLayerRects[layerIndex].z;
                    var inHeight = inputLayerRects[layerIndex].w;

                    for (var y = 0; y < inHeight; ++y)
                    {
                        var outPosY = (y + inY) - outStartY;
                        // If pixel is outside of output texture's Y, move to the next pixel.
                        if (outPosY < 0 || outPosY >= outHeight)
                            continue;
                        
                        // Flip Y position on the input texture, because
                        // PSDs textures are stored "upside-down"
                        var inRow = ((inHeight - 1) - y) * inWidth;
                        var outRow = outPosY * outWidth;

                        for (var x = 0; x < inWidth; ++x)
                        {
                            var outPosX = (x + inX) - outStartX;
                            // If pixel is outside of output texture's X, move to the next pixel.
                            if (outPosX < 0 || outPosX >= outWidth)
                                continue;

                            var inBufferIndex = inRow + x;
                            var outBufferIndex = outRow + outPosX;
                            if (outBufferIndex < 0 || outBufferIndex > (outWidth * outHeight))
                                continue;

                            Color inColor = inputColor[inBufferIndex];
                            Color prevOutColor = outputColor[outBufferIndex];
                            var outColor = new Color();

                            var destAlpha = prevOutColor.a * (1 - inColor.a);
                            outColor.a = inColor.a + prevOutColor.a * (1 - inColor.a);
                            
                            var premultiplyAlpha = outColor.a > 0.0f ? 1 / outColor.a : 1f;
                            outColor.r = (inColor.r * inColor.a + prevOutColor.r * destAlpha) * premultiplyAlpha;
                            outColor.g = (inColor.g * inColor.a + prevOutColor.g * destAlpha) * premultiplyAlpha;
                            outColor.b = (inColor.b * inColor.a + prevOutColor.b * destAlpha) * premultiplyAlpha;

                            outputColor[outBufferIndex] = outColor;
                        }
                    }
                }
            }
        }

        public static unsafe void Execute(in PSDExtractLayerData[] psdExtractLayerData, out List<PSDLayer> outputLayers, bool importHiddenLayer, Vector2Int canvasSize)
        {
            outputLayers = new List<PSDLayer>();
            UnityEngine.Profiling.Profiler.BeginSample("ExtractLayer_PrepareJob");

            var inputLayers = new List<PSDLayer>();
            ExtractLayerData(in psdExtractLayerData, ref inputLayers, importHiddenLayer, false, true, canvasSize);
            
            var layerGroupData = new List<LayerGroupData>();
            GenerateOutputLayers(in inputLayers, ref outputLayers, ref layerGroupData, false, canvasSize);

            if (layerGroupData.Count == 0)
            {
                foreach (var layer in outputLayers)
                    layer.texture = default;
                return;
            }
            
            var job = new ConvertBufferJob();
            job.inputTextureBufferSizes = new NativeArray<int>(inputLayers.Count, Allocator.TempJob);
            job.inputTextures = new NativeArray<IntPtr>(inputLayers.Count, Allocator.TempJob);
            job.inputLayerRects = new NativeArray<int4>(inputLayers.Count, Allocator.TempJob);
            job.outputLayerRect = new NativeArray<int4>(layerGroupData.Count, Allocator.TempJob);
            job.outputTextures = new NativeArray<IntPtr>(layerGroupData.Count, Allocator.TempJob);
            
            for (int i = 0, groupIndex = 0; i < inputLayers.Count; ++i)
            {
                var inputLayer = inputLayers[i];
                var outputLayer = outputLayers[i];
                
                job.inputTextures[i] = inputLayer.texture.IsCreated ? new IntPtr(inputLayer.texture.GetUnsafePtr()) : IntPtr.Zero;

                var isGroupOwner = groupIndex < layerGroupData.Count && layerGroupData[groupIndex].startIndex == i;
                if (isGroupOwner)
                {
                    outputLayer.texture = new NativeArray<Color32>(outputLayer.width * outputLayer.height, Allocator.Persistent);
                    
                    job.outputLayerRect[groupIndex] = new int4((int)outputLayer.layerPosition.x, (int)outputLayer.layerPosition.y, outputLayer.width, outputLayer.height);
                    job.outputTextures[groupIndex] = outputLayer.texture.IsCreated ? new IntPtr(outputLayer.texture.GetUnsafePtr()) : IntPtr.Zero;
                    job.inputTextureBufferSizes[i] = inputLayer.texture.IsCreated ? inputLayer.texture.Length : -1;
                    job.inputLayerRects[i] = layerGroupData[groupIndex].documentRect;
                    ++groupIndex;
                }
                else
                {
                    job.inputTextureBufferSizes[i] = inputLayer.texture.IsCreated ? inputLayer.texture.Length : -1;
                    job.inputLayerRects[i] = new int4((int)inputLayer.layerPosition.x, (int)inputLayer.layerPosition.y, inputLayer.width, inputLayer.height);
                    outputLayer.texture = default;
                }
            }
            job.layerGroupDataData = new NativeArray<LayerGroupData>(layerGroupData.ToArray(), Allocator.TempJob);

            var jobsPerThread = layerGroupData.Count / (SystemInfo.processorCount == 0 ? 8 : SystemInfo.processorCount);
            jobsPerThread = Mathf.Max(jobsPerThread, 1);
            var handle = job.Schedule(layerGroupData.Count, jobsPerThread);
            UnityEngine.Profiling.Profiler.EndSample();
            handle.Complete();
        }

        static void ExtractLayerData(in PSDExtractLayerData[] inputLayers, ref List<PSDLayer> extractedLayers, bool importHiddenLayer, bool flatten, bool parentGroupVisible, Vector2Int canvasSize)
        {
            var parentGroupIndex = extractedLayers.Count - 1;
            
            foreach (var inputLayer in inputLayers)
            {
                var bitmapLayer = inputLayer.bitmapLayer;
                var importSettings = inputLayer.importSetting;
                var layerVisible = bitmapLayer.Visible && parentGroupVisible;
                
                var layerRect = new RectInt(bitmapLayer.documentRect.X, bitmapLayer.documentRect.Y, bitmapLayer.Surface.width, bitmapLayer.Surface.height);
                
                if (!bitmapLayer.IsGroup)
                    layerRect.y = (canvasSize.y - layerRect.y - layerRect.height);

                NativeArray<Color32> surface = default;
                if ((importHiddenLayer || bitmapLayer.Visible) && 
                    importSettings.importLayer && 
                    bitmapLayer.Surface.color.IsCreated &&
                    bitmapLayer.Surface.color.Length > 0)
                    surface = bitmapLayer.Surface.color;

                var extractedLayer = new PSDLayer(surface, parentGroupIndex, bitmapLayer.IsGroup, bitmapLayer.Name, layerRect.width, layerRect.height, bitmapLayer.LayerID, bitmapLayer.Visible)
                {
                    spriteID = inputLayer.importSetting.spriteId,
                    flatten = bitmapLayer.IsGroup && inputLayer.importSetting.flatten,
                    layerPosition = bitmapLayer.IsGroup ? Vector2.zero : layerRect.position
                };
                
                extractedLayer.isImported = (importHiddenLayer || layerVisible) && !flatten && importSettings.importLayer;
                if (extractedLayer.isGroup)
                    extractedLayer.isImported = extractedLayer.isImported && extractedLayer.flatten;

                extractedLayers.Add(extractedLayer);
                
                if (inputLayer.children.Length > 0)
                    ExtractLayerData(in inputLayer.children, ref extractedLayers, importHiddenLayer, flatten || extractedLayer.flatten, layerVisible, canvasSize);
            }
        }

        static void GenerateOutputLayers(in List<PSDLayer> inputLayers, ref List<PSDLayer> outputLayers, ref List<LayerGroupData> layerGroupData, bool flatten, Vector2Int canvasSize)
        {
            var canvasRect = new RectInt(Vector2Int.zero, canvasSize);
            
            for (var i = 0; i < inputLayers.Count; ++i)
            {
                var inputLayer = inputLayers[i];
                
                var outputLayer = new PSDLayer(inputLayer);
                var outputRect = new RectInt((int)outputLayer.layerPosition.x, (int)outputLayer.layerPosition.y, outputLayer.width, outputLayer.height);
                
                if (inputLayer.isGroup)
                {
                    var childIndices = FindAllChildrenOfParent(i, in inputLayers);
                    childIndices.Sort();
                    
                    var startIndex = i;
                    var endIndex = i + childIndices.Count;
                    
                    if (flatten == false && inputLayer.flatten && startIndex < endIndex)
                    {
                        var groupBoundingBox = CalculateLayerRectInChildren(in inputLayers, in childIndices);
                        layerGroupData.Add(new LayerGroupData()
                        {
                            startIndex = startIndex,
                            endIndex = endIndex,
                            documentRect = new int4(groupBoundingBox.x, groupBoundingBox.y, groupBoundingBox.width, groupBoundingBox.height)
                        });
                        outputLayer.texture = default;
                        outputRect = groupBoundingBox;
                    }
                }
                else if(!inputLayer.isGroup && inputLayer.isImported)
                {
                    var inputRect = new int4((int)inputLayer.layerPosition.x, (int)inputLayer.layerPosition.y, inputLayer.width, inputLayer.height);
                    layerGroupData.Add(new LayerGroupData()
                    {
                        startIndex = i,
                        endIndex = i,
                        documentRect = inputRect
                    });
                }
                
                CropRect(ref outputRect, canvasRect);
                outputLayer.layerPosition = outputRect.position;
                outputLayer.width = outputRect.width;
                outputLayer.height = outputRect.height;
                
                outputLayers.Add(outputLayer);
            }
        }

        static List<int> FindAllChildrenOfParent(int parentIndex, in List<PSDLayer> layers)
        {
            var childIndices = new List<int>();
            for (var i = parentIndex + 1; i < layers.Count; ++i)
            {
                if (layers[i].parentIndex == parentIndex)
                {
                    childIndices.Add(i);
                    if (layers[i].isGroup)
                        childIndices.AddRange(FindAllChildrenOfParent(i, in layers));
                }
            }
            return childIndices;
        }
        
        static RectInt CalculateLayerRectInChildren(in List<PSDLayer> inputLayers, in List<int> childIndices)
        {
            var groupBoundingBox = default(RectInt);
            for (var m = 0; m < childIndices.Count; ++m)
            {
                var childLayer = inputLayers[childIndices[m]];
                if (childLayer.isGroup) 
                    continue;
                
                var layerRect = new RectInt((int) childLayer.layerPosition.x, (int) childLayer.layerPosition.y,
                    childLayer.width, childLayer.height);
                if (IsRectIntDefault(groupBoundingBox))
                    groupBoundingBox = layerRect;
                else
                    FitRectInsideRect(ref groupBoundingBox, in layerRect);
            }

            return groupBoundingBox;
        }        

        static bool IsRectIntDefault(RectInt rectInt)
        {
            return rectInt.x == 0 &&
                   rectInt.y == 0 &&
                   rectInt.width == 0 &&
                   rectInt.height == 0;
        }

        static void CropRect(ref RectInt baseRect, in RectInt cropArea)
        {
            if (baseRect.x < cropArea.x)
            {
                baseRect.width = Mathf.Max(baseRect.width - (cropArea.x - baseRect.x), 0);
                baseRect.x = cropArea.x;
            }
            if (baseRect.xMax > cropArea.xMax)
            {
                baseRect.x = Mathf.Min(baseRect.x, cropArea.xMax);
                baseRect.width = Mathf.Max(cropArea.xMax - baseRect.x, 0);
            }

            if (baseRect.y < cropArea.y)
            {
                baseRect.height = Mathf.Max(baseRect.height - (cropArea.y - baseRect.y), 0);
                baseRect.y = cropArea.y;
            }
            if (baseRect.yMax > cropArea.yMax)
            {
                baseRect.y = Mathf.Min(baseRect.y, cropArea.yMax);
                baseRect.height = Mathf.Max(cropArea.yMax - baseRect.y, 0);
            }
        }

        static void FitRectInsideRect(ref RectInt baseRect, in RectInt rectToFitIn)
        {
            if (baseRect.xMin > rectToFitIn.xMin)
                baseRect.xMin = rectToFitIn.xMin;
            if (baseRect.yMin > rectToFitIn.yMin)
                baseRect.yMin = rectToFitIn.yMin;
            if (baseRect.xMax < rectToFitIn.xMax)
                baseRect.xMax = rectToFitIn.xMax;
            if (baseRect.yMax < rectToFitIn.yMax)
                baseRect.yMax = rectToFitIn.yMax;            
        }
    }
}