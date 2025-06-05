using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using UnityEditor.U2D.Sprites;
using Unity.Jobs;
using Unity.Collections;

namespace UnityEditor.U2D.Animation
{
    internal struct WeightedTriangle : IComparable<WeightedTriangle>
    {
        public int p1;
        public int p2;
        public int p3;
        public float weight;

        public int CompareTo(WeightedTriangle other)
        {
            return weight.CompareTo(other.weight);
        }
    }

    internal struct SpriteJobData
    {
        public BaseSpriteMeshData spriteMesh;
        public NativeArray<float2> vertices;
        public NativeArray<int2> edges;
        public NativeArray<int> indices;
        public NativeArray<BoneWeight> weights;
        public NativeArray<int4> result;
    };

    internal class SpriteMeshDataController
    {
        public BaseSpriteMeshData spriteMeshData;
        float2[] m_VerticesTemp = new float2[0];
        int2[] m_EdgesTemp = new int2[0];

        public void CreateVertex(Vector2 position)
        {
            CreateVertex(position, -1);
        }

        public void CreateVertex(Vector2 position, int edgeIndex)
        {
            Debug.Assert(spriteMeshData != null, "Assert failed. Expected: spriteMeshData != null. Actual: spriteMeshData == null");

            spriteMeshData.AddVertex(position, default(BoneWeight));

            if (edgeIndex != -1)
            {
                var edge = spriteMeshData.edges[edgeIndex];
                RemoveEdge(edge);
                CreateEdge(edge.x, spriteMeshData.vertexCount - 1);
                CreateEdge(edge.y, spriteMeshData.vertexCount - 1);
            }
        }

        public void CreateEdge(int index1, int index2)
        {
            Debug.Assert(spriteMeshData != null, "Assert failed. Expected: spriteMeshData != null. Actual: spriteMeshData == null");
            Debug.Assert(index1 >= 0, $"Assert failed. Expected: index1 >= 0. Actual: index1 == {index1}");
            Debug.Assert(index2 >= 0, $"Assert failed. Expected: index2 >= 0. Actual: index2 == {index2}");
            Debug.Assert(index1 < spriteMeshData.vertexCount, $"Assert failed. Expected: index1 < spriteMeshData.vertexCount. Actual: index1 == {index1} spriteMeshData.vertexCount == {spriteMeshData.vertexCount}");
            Debug.Assert(index2 < spriteMeshData.vertexCount, $"Assert failed. Expected: index2 < spriteMeshData.vertexCount. Actual: index2 == {index2} spriteMeshData.vertexCount == {spriteMeshData.vertexCount}");
            Debug.Assert(index1 != index2, $"Assert failed. Expected: index1 != index2. Actual: index1 == {index1} index2 == {index2}");

            var newEdge = new int2(index1, index2);
            if (!spriteMeshData.edges.ContainsAny(newEdge))
            {
                var listOfEdges = new List<int2>(spriteMeshData.edges)
                {
                    newEdge
                };
                spriteMeshData.SetEdges(listOfEdges.ToArray());
            }
        }

        public void RemoveVertex(int index)
        {
            Debug.Assert(spriteMeshData != null);

            //We need to delete the edges that reference the index
            if (FindEdgesContainsIndex(index, out var edgesWithIndex))
            {
                //If there are 2 edges referencing the same index we are removing, we can create a new one that connects the endpoints ("Unsplit").
                if (edgesWithIndex.Count == 2)
                {
                    var first = edgesWithIndex[0];
                    var second = edgesWithIndex[1];

                    int index1 = first.x != index ? first.x : first.y;
                    int index2 = second.x != index ? second.x : second.y;

                    CreateEdge(index1, index2);
                }

                //remove found edges
                for (int i = 0; i < edgesWithIndex.Count; i++)
                {
                    RemoveEdge(edgesWithIndex[i]);
                }
            }

            //Fix indices in edges greater than the one we are removing
            for (int i = 0; i < spriteMeshData.edges.Length; i++)
            {
                var edge = spriteMeshData.edges[i];

                if (edge.x > index)
                    edge.x--;
                if (edge.y > index)
                    edge.y--;

                spriteMeshData.edges[i] = edge;
            }

            spriteMeshData.RemoveVertex(index);
        }

        public void RemoveVertex(IEnumerable<int> indices)
        {
            var sortedIndexList = new List<int>(indices);

            if (sortedIndexList.Count == 0)
                return;

            sortedIndexList.Sort();

            for (var i = sortedIndexList.Count - 1; i >= 0; --i)
            {
                RemoveVertex(sortedIndexList[i]);
            }
        }

        void RemoveEdge(int2 edge)
        {
            Debug.Assert(spriteMeshData != null);
            var listOfEdges = new List<int2>(spriteMeshData.edges);
            listOfEdges.Remove(edge);
            spriteMeshData.SetEdges(listOfEdges.ToArray());
        }

        bool FindEdgesContainsIndex(int index, out List<int2> result)
        {
            Debug.Assert(spriteMeshData != null);

            bool found = false;

            result = new List<int2>();

            for (int i = 0; i < spriteMeshData.edges.Length; ++i)
            {
                var edge = spriteMeshData.edges[i];
                if (edge.x == index || edge.y == index)
                {
                    found = true;
                    result.Add(edge);
                }
            }

            return found;
        }

        public void CreateQuad()
        {
            var frame = new Rect(Vector2.zero, spriteMeshData.frame.size);
            var verts = new Vector2[]
            {
                new Vector2(frame.xMin, frame.yMin),
                new Vector2(frame.xMax, frame.yMin),
                new Vector2(frame.xMin, frame.yMax),
                new Vector2(frame.xMax, frame.yMax)
            };

            for (var i = 0; i < verts.Length; ++i)
                CreateVertex(verts[i]);

            var tris = new int[]
            {
                0, 2, 3, 1
            };

            for (var i = 0; i < tris.Length; ++i)
            {
                var n = (i + 1) % tris.Length;
                CreateEdge(tris[i], tris[n]);
            }
        }

        public JobHandle TriangulateJob(ITriangulator triangulator, SpriteJobData spriteData)
        {
            Debug.Assert(spriteMeshData != null);
            Debug.Assert(triangulator != null);

            FillMeshDataContainers(out m_VerticesTemp, out m_EdgesTemp, out var weightData, out var hasWeightData);
            return triangulator.ScheduleTriangulate(in m_VerticesTemp, in m_EdgesTemp, ref spriteData.vertices, ref spriteData.indices, ref spriteData.edges, ref spriteData.result);
        }

        public void Triangulate(ITriangulator triangulator)
        {
            Debug.Assert(spriteMeshData != null);
            Debug.Assert(triangulator != null);

            FillMeshDataContainers(out m_VerticesTemp, out m_EdgesTemp, out var weightData, out var hasWeightData);
            triangulator.Triangulate(ref m_EdgesTemp, ref m_VerticesTemp, out var indices);

            if (m_VerticesTemp.Length == 0 || indices.Length == 0)
            {
                spriteMeshData.Clear();
                CreateQuad();

                FillMeshDataContainers(out m_VerticesTemp, out m_EdgesTemp, out weightData, out hasWeightData);
                triangulator.Triangulate(ref m_EdgesTemp, ref m_VerticesTemp, out indices);
            }

            spriteMeshData.Clear();
            spriteMeshData.SetIndices(indices);
            spriteMeshData.SetEdges(m_EdgesTemp);

            var hasNewVertices = m_VerticesTemp.Length != weightData.Length;
            for (var i = 0; i < m_VerticesTemp.Length; ++i)
            {
                var boneWeight = default(BoneWeight);
                if (!hasNewVertices)
                    boneWeight = weightData[i].ToBoneWeight(true);
                spriteMeshData.AddVertex(m_VerticesTemp[i], boneWeight);
            }

            if (hasNewVertices && hasWeightData)
                CalculateWeights(new BoundedBiharmonicWeightsGenerator(), null, 0.01f);
        }

        void FillMeshDataContainers(out float2[] vertices, out int2[] edges, out EditableBoneWeight[] weightData, out bool hasWeightData)
        {
            edges = spriteMeshData.edges;
            vertices = EditorUtilities.ToFloat2(spriteMeshData.vertices);

            weightData = new EditableBoneWeight[spriteMeshData.vertexWeights.Length];
            Array.Copy(spriteMeshData.vertexWeights, weightData, weightData.Length);

            hasWeightData = false;
            if (weightData.Length > 0 && weightData[0] != default)
                hasWeightData = true;
        }

        public JobHandle Subdivide(ITriangulator triangulator, SpriteJobData spriteData, float largestAreaFactor, float areaThreshold)
        {
            Debug.Assert(spriteMeshData != null);
            Debug.Assert(triangulator != null);

            m_EdgesTemp = spriteMeshData.edges;
            m_VerticesTemp = EditorUtilities.ToFloat2(spriteMeshData.vertices);

            try
            {
                triangulator.Tessellate(0f, 0f, 0f, largestAreaFactor, areaThreshold, 100, ref m_VerticesTemp, ref m_EdgesTemp, out var indices);

                spriteMeshData.Clear();

                for (var i = 0; i < m_VerticesTemp.Length; ++i)
                    spriteMeshData.AddVertex(m_VerticesTemp[i], default(BoneWeight));

                spriteMeshData.SetIndices(indices);
                spriteMeshData.SetEdges(m_EdgesTemp);
            }
            catch (Exception) { }

            return default(JobHandle);
        }

        public void ClearWeights(ISelection<int> selection)
        {
            Debug.Assert(spriteMeshData != null);

            for (var i = 0; i < spriteMeshData.vertexCount; ++i)
                if (selection == null || (selection.Count == 0 || selection.Contains(i)))
                    spriteMeshData.vertexWeights[i].SetFromBoneWeight(default(BoneWeight));
        }

        public void OutlineFromAlpha(IOutlineGenerator outlineGenerator, ITextureDataProvider textureDataProvider, float outlineDetail, byte alphaTolerance)
        {
            Debug.Assert(spriteMeshData != null, "Assert failed. Expected: spriteMeshData != null. Actual: spriteMeshData == null");
            Debug.Assert(textureDataProvider != null, "Assert failed. Expected: textureDataProvider != null. Actual: textureDataProvider == null");
            Debug.Assert(textureDataProvider.texture != null, "Assert failed. Expected: textureDataProvider.texture != null. Actual: textureDataProvider.texture == null");

            int width, height;
            textureDataProvider.GetTextureActualWidthAndHeight(out width, out height);

            var scale = new Vector2(textureDataProvider.texture.width / (float)width, textureDataProvider.texture.height / (float)height);
            var scaleInv = new Vector2(1f / scale.x, 1f / scale.y);
            var rectOffset = spriteMeshData.frame.size * 0.5f;

            var scaledRect = spriteMeshData.frame;
            scaledRect.min = Vector2.Scale(scaledRect.min, scale);
            scaledRect.max = Vector2.Scale(scaledRect.max, scale);

            spriteMeshData.Clear();

            outlineGenerator.GenerateOutline(textureDataProvider, scaledRect, outlineDetail, alphaTolerance, false, out var paths);

            var vertexIndexBase = 0;

            var vertices = new List<Vector2>(spriteMeshData.vertices);
            var edges = new List<int2>(spriteMeshData.edges);
            for (var i = 0; i < paths.Length; ++i)
            {
                var numPathVertices = paths[i].Length;

                for (var j = 0; j <= numPathVertices; j++)
                {
                    if (j < numPathVertices)
                        vertices.Add(Vector2.Scale(paths[i][j], scaleInv) + rectOffset);
                    if (j > 0)
                        edges.Add(new int2(vertexIndexBase + j - 1, vertexIndexBase + j % numPathVertices));
                }

                vertexIndexBase += numPathVertices;
            }

            var vertexWeights = new EditableBoneWeight[vertices.Count];
            for (var i = 0; i < vertexWeights.Length; ++i)
                vertexWeights[i] = new EditableBoneWeight();

            spriteMeshData.SetVertices(vertices.ToArray(), vertexWeights);
            spriteMeshData.SetEdges(edges.ToArray());
        }

        public void NormalizeWeights(ISelection<int> selection)
        {
            Debug.Assert(spriteMeshData != null);

            for (var i = 0; i < spriteMeshData.vertexCount; ++i)
                if (selection == null || (selection.Count == 0 || selection.Contains(i)))
                    spriteMeshData.vertexWeights[i].Normalize();
        }

        public JobHandle CalculateWeightsJob(IWeightsGenerator weightsGenerator, ISelection<int> selection, float filterTolerance, SpriteJobData sd)
        {
            Debug.Assert(spriteMeshData != null);

            GetControlPoints(out var controlPoints, out var bones, out var pins);

            var vertices = EditorUtilities.ToFloat2(spriteMeshData.vertices);
            var indices = spriteMeshData.indices;
            var edges = spriteMeshData.edges;

            return weightsGenerator.CalculateJob(spriteMeshData.spriteName, in vertices, in indices, in edges, in controlPoints, in bones, in pins, sd);
        }

        public void CalculateWeights(IWeightsGenerator weightsGenerator, ISelection<int> selection, float filterTolerance)
        {
            Debug.Assert(spriteMeshData != null);

            GetControlPoints(out var controlPoints, out var bones, out var pins);

            var vertices = EditorUtilities.ToFloat2(spriteMeshData.vertices);
            var indices = spriteMeshData.indices;
            var edges = spriteMeshData.edges;

            var boneWeights = weightsGenerator.Calculate(spriteMeshData.spriteName, in vertices, in indices, in edges, in controlPoints, in bones, in pins);

            Debug.Assert(boneWeights.Length == spriteMeshData.vertexCount);

            for (var i = 0; i < spriteMeshData.vertexCount; ++i)
            {
                if (selection == null || (selection.Count == 0 || selection.Contains(i)))
                {
                    var editableBoneWeight = EditableBoneWeightUtility.CreateFromBoneWeight(boneWeights[i]);

                    if (filterTolerance > 0f)
                    {
                        editableBoneWeight.FilterChannels(filterTolerance);
                        editableBoneWeight.Normalize();
                    }

                    spriteMeshData.vertexWeights[i] = editableBoneWeight;
                }
            }
        }

        public void CalculateWeightsSafe(IWeightsGenerator weightsGenerator, ISelection<int> selection, float filterTolerance)
        {
            var tempSelection = new IndexedSelection();
            var vertexSelector = new GenericVertexSelector();

            vertexSelector.spriteMeshData = spriteMeshData;
            vertexSelector.selection = tempSelection;
            vertexSelector.SelectionCallback = (int i) =>
            {
                return spriteMeshData.vertexWeights[i].Sum() == 0f && (selection == null || selection.Count == 0 || selection.Contains(i));
            };
            vertexSelector.Select();

            if (tempSelection.Count > 0)
                CalculateWeights(weightsGenerator, tempSelection, filterTolerance);
        }

        public void SmoothWeights(int iterations, ISelection<int> selection)
        {
            var boneWeights = new BoneWeight[spriteMeshData.vertexCount];

            for (var i = 0; i < spriteMeshData.vertexCount; i++)
                boneWeights[i] = spriteMeshData.vertexWeights[i].ToBoneWeight(false);

            SmoothingUtility.SmoothWeights(boneWeights, spriteMeshData.indices, spriteMeshData.boneCount, iterations, out var smoothedWeights);

            for (var i = 0; i < spriteMeshData.vertexCount; i++)
                if (selection == null || (selection.Count == 0 || selection.Contains(i)))
                    spriteMeshData.vertexWeights[i].SetFromBoneWeight(smoothedWeights[i]);
        }

        public bool FindTriangle(Vector2 point, out Vector3Int indices, out Vector3 barycentricCoords)
        {
            Debug.Assert(spriteMeshData != null);

            indices = Vector3Int.zero;
            barycentricCoords = Vector3.zero;

            if (spriteMeshData.indices.Length < 3)
                return false;

            var triangleCount = spriteMeshData.indices.Length / 3;

            for (var i = 0; i < triangleCount; ++i)
            {
                indices.x = spriteMeshData.indices[i * 3];
                indices.y = spriteMeshData.indices[i * 3 + 1];
                indices.z = spriteMeshData.indices[i * 3 + 2];

                MathUtility.Barycentric(
                    point,
                    spriteMeshData.vertices[indices.x],
                    spriteMeshData.vertices[indices.y],
                    spriteMeshData.vertices[indices.z],
                    out barycentricCoords);

                if (barycentricCoords.x >= 0f && barycentricCoords.y >= 0f && barycentricCoords.z >= 0f)
                    return true;
            }

            return false;
        }

        List<float> m_VertexOrderList = new List<float>(1000);
        List<WeightedTriangle> m_WeightedTriangles = new List<WeightedTriangle>(1000);

        public void SortTrianglesByDepth()
        {
            Debug.Assert(spriteMeshData != null);

            if (spriteMeshData.boneCount == 0)
                return;

            m_VertexOrderList.Clear();
            m_WeightedTriangles.Clear();

            for (var i = 0; i < spriteMeshData.vertexCount; i++)
            {
                var vertexOrder = 0f;
                var boneWeight = spriteMeshData.vertexWeights[i];

                for (var j = 0; j < boneWeight.Count; ++j)
                    vertexOrder += spriteMeshData.GetBoneDepth(boneWeight[j].boneIndex) * boneWeight[j].weight;

                m_VertexOrderList.Add(vertexOrder);
            }

            for (var i = 0; i < spriteMeshData.indices.Length; i += 3)
            {
                var p1 = spriteMeshData.indices[i];
                var p2 = spriteMeshData.indices[i + 1];
                var p3 = spriteMeshData.indices[i + 2];
                var weight = (m_VertexOrderList[p1] + m_VertexOrderList[p2] + m_VertexOrderList[p3]) / 3f;

                m_WeightedTriangles.Add(new WeightedTriangle() { p1 = p1, p2 = p2, p3 = p3, weight = weight });
            }

            m_WeightedTriangles.Sort();

            var newIndices = new int[m_WeightedTriangles.Count * 3];
            for (var i = 0; i < m_WeightedTriangles.Count; ++i)
            {
                var triangle = m_WeightedTriangles[i];
                var indexCount = i * 3;
                newIndices[indexCount] = triangle.p1;
                newIndices[indexCount + 1] = triangle.p2;
                newIndices[indexCount + 2] = triangle.p3;
            }

            spriteMeshData.SetIndices(newIndices);
        }

        public void GetMultiEditChannelData(ISelection<int> selection, int channel,
            out bool enabled, out int boneIndex, out float weight,
            out bool isEnabledMixed, out bool isBoneIndexMixed, out bool isWeightMixed)
        {
            Debug.Assert(spriteMeshData != null);

            if (selection == null)
                throw new ArgumentNullException("selection is null");

            var first = true;
            enabled = false;
            boneIndex = -1;
            weight = 0f;
            isEnabledMixed = false;
            isBoneIndexMixed = false;
            isWeightMixed = false;

            var indices = selection.elements;

            foreach (var i in indices)
            {
                var editableBoneWeight = spriteMeshData.vertexWeights[i];

                if (first)
                {
                    enabled = editableBoneWeight[channel].enabled;
                    boneIndex = editableBoneWeight[channel].boneIndex;
                    weight = editableBoneWeight[channel].weight;

                    first = false;
                }
                else
                {
                    if (enabled != editableBoneWeight[channel].enabled)
                    {
                        isEnabledMixed = true;
                        enabled = false;
                    }

                    if (boneIndex != editableBoneWeight[channel].boneIndex)
                    {
                        isBoneIndexMixed = true;
                        boneIndex = -1;
                    }

                    if (Mathf.Abs(weight - editableBoneWeight[channel].weight) > Mathf.Epsilon)
                    {
                        isWeightMixed = true;
                        weight = 0f;
                    }
                }
            }
        }

        public void SetMultiEditChannelData(ISelection<int> selection, int channel,
            bool oldEnabled, bool newEnabled, int oldBoneIndex, int newBoneIndex, float oldWeight, float newWeight)
        {
            Debug.Assert(spriteMeshData != null);

            if (selection == null)
                throw new ArgumentNullException("selection is null");

            var channelEnabledChanged = oldEnabled != newEnabled;
            var boneIndexChanged = oldBoneIndex != newBoneIndex;
            var weightChanged = Mathf.Abs(oldWeight - newWeight) > Mathf.Epsilon;

            var indices = selection.elements;

            foreach (var i in indices)
            {
                var editableBoneWeight = spriteMeshData.vertexWeights[i];

                if (channelEnabledChanged)
                    editableBoneWeight[channel].enabled = newEnabled;

                if (boneIndexChanged)
                    editableBoneWeight[channel].boneIndex = newBoneIndex;

                if (weightChanged)
                    editableBoneWeight[channel].weight = newWeight;

                if (channelEnabledChanged || weightChanged)
                    editableBoneWeight.CompensateOtherChannels(channel);
            }
        }

        public void GetControlPoints(out float2[] points, out int2[] edges, out int[] pins)
        {
            Debug.Assert(spriteMeshData != null);

            points = null;
            edges = null;

            var pointList = new List<Vector2>();
            var edgeList = new List<int2>();
            var pinList = new List<int>();
            var bones = new List<SpriteBoneData>(spriteMeshData.boneCount);

            for (var i = 0; i < spriteMeshData.boneCount; ++i)
                bones.Add(spriteMeshData.GetBoneData(i));

            foreach (var bone in bones)
            {
                var length = (bone.endPosition - bone.position).magnitude;

                if (length > 0f)
                {
                    var index1 = FindPoint(pointList, bone.position, 0.01f);
                    var index2 = FindPoint(pointList, bone.endPosition, 0.01f);

                    if (index1 == -1)
                    {
                        pointList.Add(bone.position);
                        index1 = pointList.Count - 1;
                    }

                    if (index2 == -1)
                    {
                        pointList.Add(bone.endPosition);
                        index2 = pointList.Count - 1;
                    }

                    edgeList.Add(new int2(index1, index2));
                }
                else if (bone.length == 0f)
                {
                    pointList.Add(bone.position);
                    pinList.Add(pointList.Count - 1);
                }
            }

            points = new float2[pointList.Count];
            for (var i = 0; i < pointList.Count; ++i)
                points[i] = pointList[i];

            edges = edgeList.ToArray();
            pins = pinList.ToArray();
        }

        static int FindPoint(IReadOnlyList<Vector2> points, Vector2 point, float distanceTolerance)
        {
            var sqrTolerance = distanceTolerance * distanceTolerance;
            for (var i = 0; i < points.Count; ++i)
            {
                if ((points[i] - point).sqrMagnitude <= sqrTolerance)
                    return i;
            }

            return -1;
        }

        public void SmoothFill()
        {
            var tempSelection = new IndexedSelection();
            var vertexSelector = new GenericVertexSelector();
            var currentWeightSum = 0f;
            var prevWeightSum = 0f;

            vertexSelector.spriteMeshData = spriteMeshData;
            vertexSelector.selection = tempSelection;
            vertexSelector.SelectionCallback = (int i) =>
            {
                var sum = spriteMeshData.vertexWeights[i].Sum();
                currentWeightSum += sum;
                return sum < 0.99f;
            };

            do
            {
                prevWeightSum = currentWeightSum;
                currentWeightSum = 0f;
                vertexSelector.Select();

                if (tempSelection.Count > 0)
                    SmoothWeights(1, tempSelection);
            } while (currentWeightSum - prevWeightSum > 0.001f);

            if (tempSelection.Count > 0)
                NormalizeWeights(tempSelection);
        }
    }
}
