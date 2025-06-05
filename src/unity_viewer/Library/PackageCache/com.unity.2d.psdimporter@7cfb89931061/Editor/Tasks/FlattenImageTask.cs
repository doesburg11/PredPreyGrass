using System;
using System.Collections.Generic;
using Unity.Burst;
using UnityEngine;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;

namespace UnityEditor.U2D.PSD
{
    static class FlattenImageTask
    {
        struct LayerData
        {
            public IntPtr layerBuffer;
            public int4 layerRect;
        }
        
        public static unsafe void Execute(in PSDExtractLayerData[] layer, ref NativeArray<Color32> output, bool importHiddenLayer, Vector2Int documentSize)
        {
            UnityEngine.Profiling.Profiler.BeginSample("FlattenImage");
            
            var layerData = new List<LayerData>();
            for (var i = layer.Length - 1; i >= 0; --i)
            {
                GetLayerDataToMerge(in layer[i], ref layerData, importHiddenLayer);
            }

            if (layerData.Count == 0)
                return;

            var layersPerJob = layerData.Count / (SystemInfo.processorCount == 0 ? 8 : SystemInfo.processorCount);
            layersPerJob = Mathf.Max(layersPerJob, 1);

            var job = new FlattenImageInternalJob();
            var combineJob = new FlattenImageInternalJob();

            job.inputTextures = new NativeArray<IntPtr>(layerData.Count, Allocator.TempJob);
            job.inputTextureRects = new NativeArray<int4>(layerData.Count, Allocator.TempJob);
            
            for (var i = 0; i < layerData.Count; ++i)
            {
                job.inputTextures[i] = layerData[i].layerBuffer;
                job.inputTextureRects[i] = layerData[i].layerRect;
            }

            job.layersPerJob = layersPerJob;
            job.flipY = false;
            combineJob.flipY = true;

            var jobCount = layerData.Count / layersPerJob + (layerData.Count % layersPerJob > 0 ? 1 : 0);
            combineJob.layersPerJob = jobCount;

            var premergedBuffer = new NativeArray<byte>[jobCount];
            job.outputTextureSizes = new NativeArray<int2>(jobCount, Allocator.TempJob);
            job.outputTextures = new NativeArray<IntPtr>(jobCount, Allocator.TempJob);
            combineJob.inputTextures = new NativeArray<IntPtr>(jobCount, Allocator.TempJob);
            combineJob.inputTextureRects = new NativeArray<int4>(jobCount, Allocator.TempJob);
            
            for (var i = 0; i < jobCount; ++i)
            {
                premergedBuffer[i] = new NativeArray<byte>(documentSize.x * documentSize.y * 4, Allocator.TempJob);
                job.outputTextureSizes[i] = new int2(documentSize.x, documentSize.y);
                job.outputTextures[i] = new IntPtr(premergedBuffer[i].GetUnsafePtr());
                combineJob.inputTextures[i] = new IntPtr(premergedBuffer[i].GetUnsafeReadOnlyPtr());
                combineJob.inputTextureRects[i] = new int4(0, 0, documentSize.x, documentSize.y);
            }
            
            combineJob.outputTextureSizes = new NativeArray<int2>(new [] {new int2(documentSize.x, documentSize.y) }, Allocator.TempJob);
            combineJob.outputTextures = new NativeArray<IntPtr>(new[] { new IntPtr(output.GetUnsafePtr()) }, Allocator.TempJob);

            var handle = job.Schedule(jobCount, 1);
            combineJob.Schedule(1, 1, handle).Complete();

            foreach (var b in premergedBuffer)
            {
                if (b.IsCreated)
                    b.Dispose();
            }
            
            UnityEngine.Profiling.Profiler.EndSample();
        }

        static unsafe void GetLayerDataToMerge(in PSDExtractLayerData layer, ref List<LayerData> layerData, bool importHiddenLayer)
        {
            var bitmapLayer = layer.bitmapLayer;
            var importSetting = layer.importSetting;
            if (!bitmapLayer.Visible && importHiddenLayer == false || importSetting.importLayer == false)
                return;
            
            if (bitmapLayer.IsGroup)
            {
                for (var i = layer.children.Length - 1; i >= 0; --i)
                    GetLayerDataToMerge(layer.children[i], ref layerData, importHiddenLayer);
            }

            if (bitmapLayer.Surface == null || bitmapLayer.localRect == default) 
                return;
            
            var layerRect = bitmapLayer.documentRect;
            var data = new LayerData()
            {
                layerBuffer = new IntPtr(bitmapLayer.Surface.color.GetUnsafeReadOnlyPtr()),
                layerRect = new int4(layerRect.X, layerRect.Y, layerRect.Width, layerRect.Height)
            };
            layerData.Add(data);
        }

        [BurstCompile]
        struct FlattenImageInternalJob : IJobParallelFor
        {
            [ReadOnly, DeallocateOnJobCompletion]
            public NativeArray<IntPtr> inputTextures;
            [ReadOnly, DeallocateOnJobCompletion]
            public NativeArray<int4> inputTextureRects;
            [ReadOnly]
            public int layersPerJob;
            [ReadOnly]
            public bool flipY;

            [ReadOnly, DeallocateOnJobCompletion] 
            public NativeArray<int2> outputTextureSizes;
            [DeallocateOnJobCompletion]
            public NativeArray<IntPtr> outputTextures;

            public unsafe void Execute(int index)
            {
                var outputColor = (Color32*)outputTextures[index].ToPointer();
                for (var layerIndex = index * layersPerJob; layerIndex < (index * layersPerJob) + layersPerJob; ++layerIndex)
                {
                    if (inputTextures.Length <= layerIndex)
                        break;
                    
                    var inputColor = (Color32*)inputTextures[layerIndex].ToPointer();
                    var inStartPosX = inputTextureRects[layerIndex].x;
                    var inStartPosY = inputTextureRects[layerIndex].y;
                    var inWidth = inputTextureRects[layerIndex].z;
                    var inHeight = inputTextureRects[layerIndex].w;

                    var outWidth = outputTextureSizes[index].x;
                    var outHeight = outputTextureSizes[index].y;
                    
                    for (var y = 0; y < inHeight; ++y)
                    {
                        var outPosY = y + inStartPosY;
                        // If pixel is outside of output texture's Y, move to the next pixel.
                        if (outPosY < 0 || outPosY >= outHeight)
                            continue;
                        
                        var inRow = y * inWidth;
                        var outRow = flipY ? (outHeight - 1 - y - inStartPosY) * outWidth : (y + inStartPosY) * outWidth;

                        for (var x = 0; x < inWidth; ++x)
                        {
                            var outPosX = x + inStartPosX;
                            // If pixel is outside of output texture's X, move to the next pixel.
                            if (outPosX < 0 || outPosX >= outWidth)
                                continue;
                            
                            var inBufferIndex = inRow + x;
                            var outBufferIndex = outRow + outPosX;

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
    }
}
