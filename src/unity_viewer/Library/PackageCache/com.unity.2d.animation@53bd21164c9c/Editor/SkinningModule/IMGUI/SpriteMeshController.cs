using System;
using Unity.Mathematics;
using UnityEngine;

namespace UnityEditor.U2D.Animation
{
    internal class SpriteMeshController
    {
        const float k_SnapDistance = 10f;

        struct EdgeIntersectionResult
        {
            public int startVertexIndex;
            public int endVertexIndex;
            public int intersectEdgeIndex;
            public Vector2 endPosition;
        }

        SpriteMeshDataController m_SpriteMeshDataController = new();
        EdgeIntersectionResult m_EdgeIntersectionResult;

        public ISpriteMeshView spriteMeshView { get; set; }
        public BaseSpriteMeshData spriteMeshData { get; set; }
        public ISelection<int> selection { get; set; }
        public ICacheUndo cacheUndo { get; set; }
        public ITriangulator triangulator { get; set; }

        public bool disable { get; set; }
        public Rect frame { get; set; }

        public void OnGUI()
        {
            m_SpriteMeshDataController.spriteMeshData = spriteMeshData;

            Debug.Assert(spriteMeshView != null);
            Debug.Assert(spriteMeshData != null);
            Debug.Assert(selection != null);
            Debug.Assert(cacheUndo != null);

            ValidateSelectionValues();

            spriteMeshView.selection = selection;
            spriteMeshView.frame = frame;

            EditorGUI.BeginDisabledGroup(disable);

            spriteMeshView.BeginLayout();

            if (spriteMeshView.CanLayout())
            {
                LayoutVertices();
                LayoutEdges();
            }

            spriteMeshView.EndLayout();

            if (spriteMeshView.CanRepaint())
            {
                DrawEdges();

                if (GUI.enabled)
                {
                    PreviewCreateVertex();
                    PreviewCreateEdge();
                    PreviewSplitEdge();
                }

                DrawVertices();
            }


            HandleSplitEdge();
            HandleCreateEdge();
            HandleCreateVertex();

            EditorGUI.EndDisabledGroup();

            HandleSelectVertex();
            HandleSelectEdge();

            EditorGUI.BeginDisabledGroup(disable);

            HandleMoveVertexAndEdge();

            EditorGUI.EndDisabledGroup();

            EditorGUI.BeginDisabledGroup(disable);

            HandleRemoveVertices();

            spriteMeshView.DoRepaint();

            EditorGUI.EndDisabledGroup();
        }

        void ValidateSelectionValues()
        {
            foreach (var index in selection.elements)
            {
                if (index >= spriteMeshData.vertexCount)
                {
                    selection.Clear();
                    break;
                }
            }
        }

        void LayoutVertices()
        {
            for (var i = 0; i < spriteMeshData.vertexCount; i++)
            {
                spriteMeshView.LayoutVertex(spriteMeshData.vertices[i], i);
            }
        }

        void LayoutEdges()
        {
            for (var i = 0; i < spriteMeshData.edges.Length; i++)
            {
                var edge = spriteMeshData.edges[i];
                var startPosition = spriteMeshData.vertices[edge.x];
                var endPosition = spriteMeshData.vertices[edge.y];

                spriteMeshView.LayoutEdge(startPosition, endPosition, i);
            }
        }

        void DrawEdges()
        {
            UpdateEdgeIntersection();

            spriteMeshView.BeginDrawEdges();

            for (var i = 0; i < spriteMeshData.edges.Length; ++i)
            {
                if (SkipDrawEdge(i))
                    continue;

                var edge = spriteMeshData.edges[i];
                var startPosition = spriteMeshData.vertices[edge.x];
                var endPosition = spriteMeshData.vertices[edge.y];

                if (selection.Contains(edge.x) && selection.Contains(edge.y))
                    spriteMeshView.DrawEdgeSelected(startPosition, endPosition);
                else
                    spriteMeshView.DrawEdge(startPosition, endPosition);
            }

            if (spriteMeshView.IsActionActive(MeshEditorAction.SelectEdge))
            {
                var hoveredEdge = spriteMeshData.edges[spriteMeshView.hoveredEdge];
                var startPosition = spriteMeshData.vertices[hoveredEdge.x];
                var endPosition = spriteMeshData.vertices[hoveredEdge.y];

                spriteMeshView.DrawEdgeHovered(startPosition, endPosition);
            }

            spriteMeshView.EndDrawEdges();
        }

        bool SkipDrawEdge(int edgeIndex)
        {
            if (GUI.enabled == false)
                return false;

            return edgeIndex == -1 ||
                spriteMeshView.hoveredEdge == edgeIndex && spriteMeshView.IsActionActive(MeshEditorAction.SelectEdge) ||
                spriteMeshView.hoveredEdge == edgeIndex && spriteMeshView.IsActionActive(MeshEditorAction.CreateVertex) ||
                spriteMeshView.closestEdge == edgeIndex && spriteMeshView.IsActionActive(MeshEditorAction.SplitEdge) ||
                edgeIndex == m_EdgeIntersectionResult.intersectEdgeIndex && spriteMeshView.IsActionActive(MeshEditorAction.CreateEdge);
        }

        void PreviewCreateVertex()
        {
            if (spriteMeshView.mode == SpriteMeshViewMode.CreateVertex &&
                spriteMeshView.IsActionActive(MeshEditorAction.CreateVertex))
            {
                var clampedMousePos = ClampToFrame(spriteMeshView.mouseWorldPosition);

                if (spriteMeshView.hoveredEdge != -1)
                {
                    var edge = spriteMeshData.edges[spriteMeshView.hoveredEdge];

                    spriteMeshView.BeginDrawEdges();

                    spriteMeshView.DrawEdge(spriteMeshData.vertices[edge.x], clampedMousePos);
                    spriteMeshView.DrawEdge(spriteMeshData.vertices[edge.y], clampedMousePos);

                    spriteMeshView.EndDrawEdges();
                }

                spriteMeshView.DrawVertex(clampedMousePos);
            }
        }

        void PreviewCreateEdge()
        {
            if (!spriteMeshView.IsActionActive(MeshEditorAction.CreateEdge))
                return;

            spriteMeshView.BeginDrawEdges();

            spriteMeshView.DrawEdge(spriteMeshData.vertices[m_EdgeIntersectionResult.startVertexIndex], m_EdgeIntersectionResult.endPosition);

            if (m_EdgeIntersectionResult.intersectEdgeIndex != -1)
            {
                var intersectingEdge = spriteMeshData.edges[m_EdgeIntersectionResult.intersectEdgeIndex];
                spriteMeshView.DrawEdge(spriteMeshData.vertices[intersectingEdge.x], m_EdgeIntersectionResult.endPosition);
                spriteMeshView.DrawEdge(spriteMeshData.vertices[intersectingEdge.y], m_EdgeIntersectionResult.endPosition);
            }

            spriteMeshView.EndDrawEdges();

            if (m_EdgeIntersectionResult.endVertexIndex == -1)
                spriteMeshView.DrawVertex(m_EdgeIntersectionResult.endPosition);
        }

        void PreviewSplitEdge()
        {
            if (!spriteMeshView.IsActionActive(MeshEditorAction.SplitEdge))
                return;

            var clampedMousePos = ClampToFrame(spriteMeshView.mouseWorldPosition);

            var closestEdge = spriteMeshData.edges[spriteMeshView.closestEdge];

            spriteMeshView.BeginDrawEdges();

            spriteMeshView.DrawEdge(spriteMeshData.vertices[closestEdge.x], clampedMousePos);
            spriteMeshView.DrawEdge(spriteMeshData.vertices[closestEdge.y], clampedMousePos);

            spriteMeshView.EndDrawEdges();

            spriteMeshView.DrawVertex(clampedMousePos);
        }

        void DrawVertices()
        {
            for (var i = 0; i < spriteMeshData.vertexCount; i++)
            {
                var position = spriteMeshData.vertices[i];

                if (selection.Contains(i))
                    spriteMeshView.DrawVertexSelected(position);
                else if (i == spriteMeshView.hoveredVertex && spriteMeshView.IsActionHot(MeshEditorAction.None))
                    spriteMeshView.DrawVertexHovered(position);
                else
                    spriteMeshView.DrawVertex(position);
            }
        }

        void HandleSelectVertex()
        {
            if (spriteMeshView.DoSelectVertex(out var additive))
                SelectVertex(spriteMeshView.hoveredVertex, additive);
        }

        void HandleSelectEdge()
        {
            if (spriteMeshView.DoSelectEdge(out var additive))
                SelectEdge(spriteMeshView.hoveredEdge, additive);
        }

        void HandleMoveVertexAndEdge()
        {
            if (selection.Count == 0)
                return;

            if (spriteMeshView.DoMoveVertex(out var finalDeltaPos) || spriteMeshView.DoMoveEdge(out finalDeltaPos))
            {
                var selectionArray = selection.elements;

                finalDeltaPos = MathUtility.MoveRectInsideFrame(CalculateRectFromSelection(), frame, finalDeltaPos);
                var movedVertexSelection = GetMovedVertexSelection(in selectionArray, spriteMeshData.vertices, finalDeltaPos);

                if (IsMovedEdgeIntersectingWithOtherEdge(in selectionArray, in movedVertexSelection, spriteMeshData.edges, spriteMeshData.vertices))
                    return;
                if (IsMovedVertexIntersectingWithOutline(in selectionArray, in movedVertexSelection, spriteMeshData.outlineEdges, spriteMeshData.vertices))
                    return;

                cacheUndo.BeginUndoOperation(TextContent.moveVertices);
                MoveSelectedVertices(in movedVertexSelection);
            }
        }

        void HandleCreateVertex()
        {
            if (spriteMeshView.DoCreateVertex())
            {
                var position = ClampToFrame(spriteMeshView.mouseWorldPosition);
                var edgeIndex = spriteMeshView.hoveredEdge;
                if (spriteMeshView.hoveredEdge != -1)
                    CreateVertex(position, edgeIndex);
                else if (m_SpriteMeshDataController.FindTriangle(position, out var indices, out var barycentricCoords))
                    CreateVertex(position, indices, barycentricCoords);
            }
        }

        void HandleSplitEdge()
        {
            if (spriteMeshView.DoSplitEdge())
                SplitEdge(ClampToFrame(spriteMeshView.mouseWorldPosition), spriteMeshView.closestEdge);
        }

        void HandleCreateEdge()
        {
            if (spriteMeshView.DoCreateEdge())
            {
                var clampedMousePosition = ClampToFrame(spriteMeshView.mouseWorldPosition);
                var edgeIntersectionResult = CalculateEdgeIntersection(selection.activeElement, spriteMeshView.hoveredVertex, spriteMeshView.hoveredEdge, clampedMousePosition);

                if (edgeIntersectionResult.endVertexIndex != -1)
                {
                    CreateEdge(selection.activeElement, edgeIntersectionResult.endVertexIndex);
                }
                else
                {
                    if (edgeIntersectionResult.intersectEdgeIndex != -1)
                    {
                        CreateVertex(edgeIntersectionResult.endPosition, edgeIntersectionResult.intersectEdgeIndex);
                        CreateEdge(selection.activeElement, spriteMeshData.vertexCount - 1);
                    }
                    else if (m_SpriteMeshDataController.FindTriangle(edgeIntersectionResult.endPosition, out var indices, out var barycentricCoords))
                    {
                        CreateVertex(edgeIntersectionResult.endPosition, indices, barycentricCoords);
                        CreateEdge(selection.activeElement, spriteMeshData.vertexCount - 1);
                    }
                }
            }
        }

        void HandleRemoveVertices()
        {
            if (spriteMeshView.DoRemove())
                RemoveSelectedVertices();
        }

        void CreateVertex(Vector2 position, Vector3Int indices, Vector3 barycentricCoords)
        {
            var bw1 = spriteMeshData.vertexWeights[indices.x];
            var bw2 = spriteMeshData.vertexWeights[indices.y];
            var bw3 = spriteMeshData.vertexWeights[indices.z];

            var result = new EditableBoneWeight();

            foreach (var channel in bw1)
            {
                if (!channel.enabled)
                    continue;

                var weight = channel.weight * barycentricCoords.x;
                if (weight > 0f)
                    result.AddChannel(channel.boneIndex, weight, true);
            }

            foreach (var channel in bw2)
            {
                if (!channel.enabled)
                    continue;

                var weight = channel.weight * barycentricCoords.y;
                if (weight > 0f)
                    result.AddChannel(channel.boneIndex, weight, true);
            }

            foreach (var channel in bw3)
            {
                if (!channel.enabled)
                    continue;

                var weight = channel.weight * barycentricCoords.z;
                if (weight > 0f)
                    result.AddChannel(channel.boneIndex, weight, true);
            }

            result.UnifyChannelsWithSameBoneIndex();
            result.FilterChannels(0f);
            result.Clamp(4, true);

            var boneWeight = result.ToBoneWeight(true);

            cacheUndo.BeginUndoOperation(TextContent.createVertex);

            m_SpriteMeshDataController.CreateVertex(position, -1);
            spriteMeshData.vertexWeights[spriteMeshData.vertexCount - 1].SetFromBoneWeight(boneWeight);
            Triangulate();
        }

        void CreateVertex(Vector2 position, int edgeIndex)
        {
            var edge = spriteMeshData.edges[edgeIndex];
            var pos1 = spriteMeshData.vertices[edge.x];
            var pos2 = spriteMeshData.vertices[edge.y];
            var dir1 = (position - pos1);
            var dir2 = (pos2 - pos1);
            var t = Vector2.Dot(dir1, dir2.normalized) / dir2.magnitude;
            t = Mathf.Clamp01(t);
            var bw1 = spriteMeshData.vertexWeights[edge.x].ToBoneWeight(true);
            var bw2 = spriteMeshData.vertexWeights[edge.y].ToBoneWeight(true);

            var boneWeight = EditableBoneWeightUtility.Lerp(bw1, bw2, t);

            cacheUndo.BeginUndoOperation(TextContent.createVertex);

            m_SpriteMeshDataController.CreateVertex(position, edgeIndex);
            spriteMeshData.vertexWeights[spriteMeshData.vertexCount - 1].SetFromBoneWeight(boneWeight);
            Triangulate();
        }

        void SelectVertex(int index, bool additiveToggle)
        {
            if (index < 0)
                throw new ArgumentException("Index out of range");

            var selected = selection.Contains(index);
            if (selected)
            {
                if (additiveToggle)
                {
                    cacheUndo.BeginUndoOperation(TextContent.selection);
                    selection.Select(index, false);
                }
            }
            else
            {
                cacheUndo.BeginUndoOperation(TextContent.selection);

                if (!additiveToggle)
                    ClearSelection();

                selection.Select(index, true);
            }

            cacheUndo.IncrementCurrentGroup();
        }

        void SelectEdge(int index, bool additiveToggle)
        {
            Debug.Assert(index >= 0);

            var edge = spriteMeshData.edges[index];

            cacheUndo.BeginUndoOperation(TextContent.selection);

            var selected = selection.Contains(edge.x) && selection.Contains(edge.y);
            if (selected)
            {
                if (additiveToggle)
                {
                    selection.Select(edge.x, false);
                    selection.Select(edge.y, false);
                }
            }
            else
            {
                if (!additiveToggle)
                    ClearSelection();

                selection.Select(edge.x, true);
                selection.Select(edge.y, true);
            }

            cacheUndo.IncrementCurrentGroup();
        }

        void ClearSelection()
        {
            cacheUndo.BeginUndoOperation(TextContent.selection);
            selection.Clear();
        }

        void MoveSelectedVertices(in Vector2[] movedVertices)
        {
            for (var i = 0; i < selection.Count; ++i)
            {
                var index = selection.elements[i];
                spriteMeshData.vertices[index] = movedVertices[i];
            }

            Triangulate();
        }

        void CreateEdge(int fromVertexIndex, int toVertexIndex)
        {
            cacheUndo.BeginUndoOperation(TextContent.createEdge);

            m_SpriteMeshDataController.CreateEdge(fromVertexIndex, toVertexIndex);
            Triangulate();
            ClearSelection();
            selection.Select(toVertexIndex, true);

            cacheUndo.IncrementCurrentGroup();
        }

        void SplitEdge(Vector2 position, int edgeIndex)
        {
            cacheUndo.BeginUndoOperation(TextContent.splitEdge);

            CreateVertex(position, edgeIndex);

            cacheUndo.IncrementCurrentGroup();
        }

        bool IsEdgeSelected()
        {
            if (selection.Count != 2)
                return false;

            var indices = selection.elements;

            var index1 = indices[0];
            var index2 = indices[1];

            var edge = new int2(index1, index2);
            return spriteMeshData.edges.ContainsAny(edge);
        }

        void RemoveSelectedVertices()
        {
            cacheUndo.BeginUndoOperation(IsEdgeSelected() ? TextContent.removeEdge : TextContent.removeVertices);

            var verticesToRemove = selection.elements;

            var noOfVertsToDelete = verticesToRemove.Length;
            var noOfVertsInMesh = m_SpriteMeshDataController.spriteMeshData.vertexCount;
            var shouldClearMesh = (noOfVertsInMesh - noOfVertsToDelete) < 3;

            if (shouldClearMesh)
            {
                m_SpriteMeshDataController.spriteMeshData.Clear();
                m_SpriteMeshDataController.CreateQuad();
            }
            else
                m_SpriteMeshDataController.RemoveVertex(verticesToRemove);

            Triangulate();

            selection.Clear();
        }

        void Triangulate()
        {
            m_SpriteMeshDataController.Triangulate(triangulator);
            m_SpriteMeshDataController.SortTrianglesByDepth();
        }

        Vector2 ClampToFrame(Vector2 position)
        {
            return MathUtility.ClampPositionToRect(position, frame);
        }

        Rect CalculateRectFromSelection()
        {
            var rect = new Rect();

            var min = new Vector2(float.MaxValue, float.MaxValue);
            var max = new Vector2(float.MinValue, float.MinValue);

            var indices = selection.elements;

            foreach (var index in indices)
            {
                var v = spriteMeshData.vertices[index];

                min.x = Mathf.Min(min.x, v.x);
                min.y = Mathf.Min(min.y, v.y);

                max.x = Mathf.Max(max.x, v.x);
                max.y = Mathf.Max(max.y, v.y);
            }

            rect.min = min;
            rect.max = max;

            return rect;
        }

        void UpdateEdgeIntersection()
        {
            if (selection.Count == 1)
                m_EdgeIntersectionResult = CalculateEdgeIntersection(selection.activeElement, spriteMeshView.hoveredVertex, spriteMeshView.hoveredEdge, ClampToFrame(spriteMeshView.mouseWorldPosition));
        }

        EdgeIntersectionResult CalculateEdgeIntersection(int vertexIndex, int hoveredVertexIndex, int hoveredEdgeIndex, Vector2 targetPosition)
        {
            Debug.Assert(vertexIndex >= 0);

            var edgeIntersection = new EdgeIntersectionResult
            {
                startVertexIndex = vertexIndex,
                endVertexIndex = hoveredVertexIndex,
                endPosition = targetPosition,
                intersectEdgeIndex = -1
            };

            var startPoint = spriteMeshData.vertices[edgeIntersection.startVertexIndex];

            var intersectsEdge = false;
            var lastIntersectingEdgeIndex = -1;

            do
            {
                lastIntersectingEdgeIndex = edgeIntersection.intersectEdgeIndex;

                if (intersectsEdge)
                {
                    var dir = edgeIntersection.endPosition - startPoint;
                    edgeIntersection.endPosition += dir.normalized * 10f;
                }

                intersectsEdge = SegmentIntersectsEdge(startPoint, edgeIntersection.endPosition, vertexIndex, ref edgeIntersection.endPosition, out edgeIntersection.intersectEdgeIndex);

                //if we are hovering a vertex and intersect an edge indexing it we forget about the intersection
                var edges = spriteMeshData.edges;
                var edge = intersectsEdge ? edges[edgeIntersection.intersectEdgeIndex] : default;
                if (intersectsEdge && (edge.x == edgeIntersection.endVertexIndex || edge.y == edgeIntersection.endVertexIndex))
                {
                    edgeIntersection.intersectEdgeIndex = -1;
                    intersectsEdge = false;
                    edgeIntersection.endPosition = spriteMeshData.vertices[edgeIntersection.endVertexIndex];
                }

                if (intersectsEdge)
                {
                    edgeIntersection.endVertexIndex = -1;

                    var intersectingEdge = spriteMeshData.edges[edgeIntersection.intersectEdgeIndex];
                    var newPointScreen = spriteMeshView.WorldToScreen(edgeIntersection.endPosition);
                    var edgeV1 = spriteMeshView.WorldToScreen(spriteMeshData.vertices[intersectingEdge.x]);
                    var edgeV2 = spriteMeshView.WorldToScreen(spriteMeshData.vertices[intersectingEdge.y]);

                    if ((newPointScreen - edgeV1).magnitude <= k_SnapDistance)
                        edgeIntersection.endVertexIndex = intersectingEdge.x;
                    else if ((newPointScreen - edgeV2).magnitude <= k_SnapDistance)
                        edgeIntersection.endVertexIndex = intersectingEdge.y;

                    if (edgeIntersection.endVertexIndex != -1)
                    {
                        edgeIntersection.intersectEdgeIndex = -1;
                        intersectsEdge = false;
                        edgeIntersection.endPosition = spriteMeshData.vertices[edgeIntersection.endVertexIndex];
                    }
                }
            } while (intersectsEdge && lastIntersectingEdgeIndex != edgeIntersection.intersectEdgeIndex);

            edgeIntersection.intersectEdgeIndex = intersectsEdge ? edgeIntersection.intersectEdgeIndex : hoveredEdgeIndex;

            if (edgeIntersection.endVertexIndex != -1 && !intersectsEdge)
                edgeIntersection.endPosition = spriteMeshData.vertices[edgeIntersection.endVertexIndex];

            return edgeIntersection;
        }

        bool SegmentIntersectsEdge(Vector2 p1, Vector2 p2, int ignoreIndex, ref Vector2 point, out int intersectingEdgeIndex)
        {
            intersectingEdgeIndex = -1;

            var sqrDistance = float.MaxValue;

            for (var i = 0; i < spriteMeshData.edges.Length; i++)
            {
                var edge = spriteMeshData.edges[i];
                var v1 = spriteMeshData.vertices[edge.x];
                var v2 = spriteMeshData.vertices[edge.y];
                var pointTmp = Vector2.zero;

                if (edge.x != ignoreIndex && edge.y != ignoreIndex &&
                    MathUtility.SegmentIntersection(p1, p2, v1, v2, ref pointTmp))
                {
                    var sqrMagnitude = (pointTmp - p1).sqrMagnitude;
                    if (sqrMagnitude < sqrDistance)
                    {
                        sqrDistance = sqrMagnitude;
                        intersectingEdgeIndex = i;
                        point = pointTmp;
                    }
                }
            }

            return intersectingEdgeIndex != -1;
        }


        static Vector2[] GetMovedVertexSelection(in int[] selection, in Vector2[] vertices, Vector2 deltaPosition)
        {
            var movedVertices = new Vector2[selection.Length];
            for (var i = 0; i < selection.Length; i++)
            {
                var index = selection[i];
                movedVertices[i] = vertices[index] + deltaPosition;
            }

            return movedVertices;
        }

        static bool IsMovedEdgeIntersectingWithOtherEdge(in int[] selection, in Vector2[] movedVertices, in int2[] meshEdges, in Vector2[] meshVertices)
        {
            var edgeCount = meshEdges.Length;
            var edgeIntersectionPoint = Vector2.zero;

            for (var i = 0; i < edgeCount; i++)
            {
                var selectionIndex = FindSelectionIndexFromEdge(selection, meshEdges[i]);
                if (selectionIndex.x == -1 && selectionIndex.y == -1)
                    continue;

                var edgeStart = selectionIndex.x != -1 ? movedVertices[selectionIndex.x] : meshVertices[meshEdges[i].x];
                var edgeEnd = selectionIndex.y != -1 ? movedVertices[selectionIndex.y] : meshVertices[meshEdges[i].y];

                for (var o = 0; o < edgeCount; o++)
                {
                    if (o == i)
                        continue;

                    if (meshEdges[i].x == meshEdges[o].x || meshEdges[i].y == meshEdges[o].x ||
                        meshEdges[i].x == meshEdges[o].y || meshEdges[i].y == meshEdges[o].y)
                        continue;

                    var otherSelectionIndex = FindSelectionIndexFromEdge(in selection, meshEdges[o]);
                    var otherEdgeStart = otherSelectionIndex.x != -1 ? movedVertices[otherSelectionIndex.x] : meshVertices[meshEdges[o].x];
                    var otherEdgeEnd = otherSelectionIndex.y != -1 ? movedVertices[otherSelectionIndex.y] : meshVertices[meshEdges[o].y];

                    if (MathUtility.SegmentIntersection(edgeStart, edgeEnd, otherEdgeStart, otherEdgeEnd, ref edgeIntersectionPoint))
                        return true;
                }
            }

            return false;
        }

        static int2 FindSelectionIndexFromEdge(in int[] selection, int2 edge)
        {
            var selectionIndex = new int2(-1, -1);
            for (var m = 0; m < selection.Length; ++m)
            {
                if (selection[m] == edge.x)
                {
                    selectionIndex.x = m;
                    break;
                }

                if (selection[m] == edge.y)
                {
                    selectionIndex.y = m;
                    break;
                }
            }

            return selectionIndex;
        }

        static bool IsMovedVertexIntersectingWithOutline(in int[] selection, in Vector2[] movedVertices, in int2[] outlineEdges, in Vector2[] meshVertices)
        {
            var edgeIntersectionPoint = Vector2.zero;

            for (var i = 0; i < selection.Length; ++i)
            {
                var edgeStart = meshVertices[selection[i]];
                var edgeEnd = movedVertices[i];

                for (var m = 0; m < outlineEdges.Length; ++m)
                {
                    if (selection[i] == outlineEdges[m].x || selection[i] == outlineEdges[m].y)
                        continue;

                    var otherSelectionIndex = FindSelectionIndexFromEdge(in selection, outlineEdges[m]);
                    var otherEdgeStart = otherSelectionIndex.x != -1 ? movedVertices[otherSelectionIndex.x] : meshVertices[outlineEdges[m].x];
                    var otherEdgeEnd = otherSelectionIndex.y != -1 ? movedVertices[otherSelectionIndex.y] : meshVertices[outlineEdges[m].y];

                    if (MathUtility.SegmentIntersection(edgeStart, edgeEnd, otherEdgeStart, otherEdgeEnd, ref edgeIntersectionPoint))
                        return true;
                }
            }

            return false;
        }
    }
}