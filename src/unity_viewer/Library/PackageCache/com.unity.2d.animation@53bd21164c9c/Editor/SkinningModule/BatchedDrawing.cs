using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEditor.U2D.Common;
using UnityEngine;
using UnityEngine.U2D.Animation;

namespace UnityEditor.U2D.Animation
{
    [BurstCompile]
    internal static class BatchedDrawing
    {
        class Batch
        {
            public UnsafeList<float3> vertices;
            public UnsafeList<Color> vertexColors;
            public UnsafeList<int> indices;

            public Batch()
            {
                vertices = new UnsafeList<float3>(1, Allocator.Persistent);
                indices = new UnsafeList<int>(1, Allocator.Persistent);
                vertexColors = new UnsafeList<Color>(1, Allocator.Persistent);
            }

            ~Batch()
            {
                vertices.Dispose();
                indices.Dispose();
                vertexColors.Dispose();
            }

            public void Clear()
            {
                vertices.Clear();
                indices.Clear();
                vertexColors.Clear();
            }
        }

        // Unity's max index limit for meshes.
        const int k_MaxIndexLimit = 65535;

        static readonly int s_HandleSize = Shader.PropertyToID("_HandleSize");

        static readonly List<Batch> s_Batches = new List<Batch>(1) { new Batch() };
        static Mesh s_Mesh;
        static NativeArray<float3> s_VertexTmpCache = default;

        public static unsafe void RegisterLine(float3 p1, float3 p2, float3 normal, float widthP1, float widthP2, Color color)
        {
            var up = math.cross(normal, p2 - p1);
            up = math.normalize(up);

            const int dataToAdd = 6;
            var batch = GetBatch(dataToAdd);
            var startIndex = batch.vertices.Length;

            batch.indices.Resize(startIndex + dataToAdd);
            batch.vertexColors.Resize(startIndex + dataToAdd);
            batch.vertices.Resize(startIndex + dataToAdd);

            var vertexPtr = batch.vertices.Ptr;
            vertexPtr[startIndex] = p1 + up * (widthP1 * 0.5f);
            vertexPtr[startIndex + 1] = p1 - up * (widthP1 * 0.5f);
            vertexPtr[startIndex + 2] = p2 - up * (widthP2 * 0.5f);
            vertexPtr[startIndex + 3] = p1 + up * (widthP1 * 0.5f);
            vertexPtr[startIndex + 4] = p2 - up * (widthP2 * 0.5f);
            vertexPtr[startIndex + 5] = p2 + up * (widthP2 * 0.5f);

            for (var i = 0; i < dataToAdd; ++i)
            {
                batch.indices.Ptr[startIndex + i] = startIndex + i;
                batch.vertexColors.Ptr[startIndex + i] = color;
            }
        }

        public static void RegisterSolidDisc(float3 center, float3 normal, float radius, Color color)
        {
            var from = math.cross(normal, math.up());
            if (math.lengthsq(from) < 1.0 / 1000.0)
                from = math.cross(normal, math.right());
            RegisterSolidArc(center, normal, from, 360f, radius, color);
        }

        public static unsafe void RegisterSolidArc(float3 center, float3 normal, float3 from, float angle, float radius, Color color, int numSamples = 60)
        {
            numSamples = math.clamp(numSamples, 3, 60);

            if (s_VertexTmpCache == default)
                s_VertexTmpCache = new NativeArray<float3>(60, Allocator.Persistent);
            SetDiscSectionPoints(ref s_VertexTmpCache, numSamples, in normal, in from, angle);

            var dataToAdd = (numSamples - 1) * 3;
            var batch = GetBatch(dataToAdd);
            var startIndex = batch.vertices.Length;

            batch.indices.Resize(startIndex + dataToAdd);
            batch.vertexColors.Resize(startIndex + dataToAdd);
            batch.vertices.Resize(startIndex + dataToAdd);

            CreateSolidArcVertices(ref batch.vertices, startIndex, in s_VertexTmpCache, in center, numSamples, radius);

            for (var i = 0; i < dataToAdd; ++i)
            {
                batch.indices.Ptr[startIndex + i] = startIndex + i;
                batch.vertexColors.Ptr[startIndex + i] = color;
            }
        }

        [BurstCompile]
        static void CreateSolidArcVertices(
            ref UnsafeList<float3> vertexPtr,
            int startIndex,
            in NativeArray<float3> vertexCache,
            in float3 center,
            int numSamples,
            float radius)
        {
            var count = 0;
            for (var i = 1; i < numSamples; i++, count += 3)
            {
                var index = startIndex + count;
                vertexPtr[index] = center;
                vertexPtr[index + 1] = center + vertexCache[i - 1] * radius;
                vertexPtr[index + 2] = center + vertexCache[i] * radius;
            }
        }

        public static unsafe void RegisterSolidArcWithOutline(float3 center, float3 normal, float3 from, float angle,
            float radius, float outlineScale, Color color, int numSamples = 60)
        {
            numSamples = Mathf.Clamp(numSamples, 3, 60);

            if (s_VertexTmpCache == default)
                s_VertexTmpCache = new NativeArray<float3>(60, Allocator.Persistent);
            SetDiscSectionPoints(ref s_VertexTmpCache, numSamples, in normal, in from, angle);

            var dataToAdd = (numSamples - 1) * 6;
            var batch = GetBatch(dataToAdd);
            var startIndex = batch.vertices.Length;

            batch.indices.Resize(startIndex + dataToAdd);
            batch.vertexColors.Resize(startIndex + dataToAdd);
            batch.vertices.Resize(startIndex + dataToAdd);

            // var vertexPtr = batch.vertices.Ptr + startIndex;
            CreateSolidArcWithOutlineVertices(ref batch.vertices, startIndex, in s_VertexTmpCache, in center, numSamples, outlineScale, radius);

            for (var i = 0; i < dataToAdd; ++i)
            {
                batch.indices.Ptr[startIndex + i] = startIndex + i;
                batch.vertexColors.Ptr[startIndex + i] = color;
            }
        }

        [BurstCompile]
        static void CreateSolidArcWithOutlineVertices(
            ref UnsafeList<float3> vertexPtr,
            int startIndex,
            in NativeArray<float3> vertexCache,
            in float3 center,
            int numSamples,
            float outlineScale,
            float radius)
        {
            var count = 0;
            for (var i = 1; i < numSamples; i++, count += 6)
            {
                var index = startIndex + count;
                vertexPtr[index] = center + vertexCache[i - 1] * (radius * outlineScale);
                vertexPtr[index + 1] = center + vertexCache[i - 1] * radius;
                vertexPtr[index + 2] = center + vertexCache[i] * radius;
                vertexPtr[index + 3] = center + vertexCache[i - 1] * (radius * outlineScale);
                vertexPtr[index + 4] = center + vertexCache[i] * radius;
                vertexPtr[index + 5] = center + vertexCache[i] * (radius * outlineScale);
            }
        }

        [BurstCompile]
        static void SetDiscSectionPoints(ref NativeArray<float3> dest, int count, in float3 normal, in float3 from, float angle)
        {
            var angleInRadians = math.degrees(angle / (float)(count - 1));
            var rotation = quaternion.AxisAngle(normal, angleInRadians);

            var vector = math.normalize(from);
            for (var i = 0; i < count; i++)
            {
                dest[i] = vector;
                vector = math.mul(rotation, vector);
            }
        }

        static Batch GetBatch(int dataToAdd)
        {
            for (var i = 0; i < s_Batches.Count; ++i)
            {
                if ((s_Batches[i].indices.Length + dataToAdd) < k_MaxIndexLimit)
                    return s_Batches[i];
            }

            var newBatch = new Batch();
            s_Batches.Add(newBatch);
            return newBatch;
        }

        public static void Draw()
        {
            if (s_Batches[0].indices.Length == 0)
                return;
            if (Event.current.type != EventType.Repaint)
                return;

            Shader.SetGlobalFloat(s_HandleSize, 1);
            InternalEditorBridge.ApplyWireMaterial();

            for (var i = 0; i < s_Batches.Count; ++i)
            {
                DrawBatch(s_Batches[i]);
                s_Batches[i].Clear();
            }

            s_VertexTmpCache.DisposeIfCreated();
            s_VertexTmpCache = default;
        }

        static unsafe void DrawBatch(Batch batch)
        {
            var vertexPtr = batch.vertices.Ptr;
            var indexPtr = batch.indices.Ptr;
            var vertexColorPtr = batch.vertexColors.Ptr;

            var vertexCount = batch.vertices.Length;

            var vertexArr = NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<Vector3>(vertexPtr, vertexCount, batch.vertices.Allocator.ToAllocator);
            var indexArr = NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<int>(indexPtr, vertexCount, batch.indices.Allocator.ToAllocator);
            var colorArr = NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<Color>(vertexColorPtr, vertexCount, batch.vertexColors.Allocator.ToAllocator);

            NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref vertexArr, AtomicSafetyHandle.GetTempUnsafePtrSliceHandle());
            NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref indexArr, AtomicSafetyHandle.GetTempUnsafePtrSliceHandle());
            NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref colorArr, AtomicSafetyHandle.GetTempUnsafePtrSliceHandle());

            if (s_Mesh == null)
                s_Mesh = new Mesh();
            else
                s_Mesh.Clear();

            s_Mesh.SetVertices(vertexArr);
            s_Mesh.SetIndices(indexArr, MeshTopology.Triangles, 0);
            s_Mesh.SetColors(colorArr);
            Graphics.DrawMeshNow(s_Mesh, Handles.matrix);
        }
    }
}