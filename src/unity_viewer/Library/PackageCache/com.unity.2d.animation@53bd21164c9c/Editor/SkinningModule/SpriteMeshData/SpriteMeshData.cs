using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.U2D.Animation;

namespace UnityEditor.U2D.Animation
{
    [Serializable]
    internal class SpriteBoneData
    {
        public int parentId = -1;
        public Vector2 localPosition;
        public Quaternion localRotation = Quaternion.identity;
        public Vector2 position;
        public Vector2 endPosition;
        public float depth;
        public float length;
    }

    [Serializable]
    internal abstract class BaseSpriteMeshData
    {
        [SerializeField]
        Vector2[] m_Vertices = new Vector2[0];
        [SerializeField]
        EditableBoneWeight[] m_VertexWeights = new EditableBoneWeight[0];
        [SerializeField]
        int[] m_Indices = new int[0];
        [SerializeField]
        int2[] m_Edges = new int2[0];
        [SerializeField]
        int2[] m_OutlineEdges = new int2[0];

        Vector2[] m_VertexPositionsOverride = null;

        public abstract Rect frame { get; }

        public Vector2[] vertices => m_VertexPositionsOverride ?? m_Vertices;
        public EditableBoneWeight[] vertexWeights => m_VertexWeights;

        public int[] indices => m_Indices;

        public int2[] edges => m_Edges;
        public int2[] outlineEdges => m_OutlineEdges;

        public int vertexCount => m_Vertices.Length;
        public virtual int boneCount => 0;
        public virtual string spriteName => "";

        public void SetIndices(int[] newIndices)
        {
            m_Indices = newIndices;
            UpdateOutlineEdges();
        }

        void UpdateOutlineEdges()
        {
            var indicesNativeArr = new NativeArray<ushort>(m_Indices.Length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            for (var i = 0; i < indicesNativeArr.Length; ++i)
                indicesNativeArr[i] = (ushort)m_Indices[i];

            var outlineNativeArr = MeshUtilities.GetOutlineEdges(indicesNativeArr);
            m_OutlineEdges = outlineNativeArr.Length > 0 ? outlineNativeArr.ToArray() : new int2[0];

            outlineNativeArr.Dispose();
            indicesNativeArr.Dispose();
        }

        public void SetEdges(int2[] newEdges)
        {
            m_Edges = newEdges;
        }

        public void SetVertices(Vector2[] newVertices, EditableBoneWeight[] newWeights)
        {
            ClearVertexPositionOverride();

            m_Vertices = newVertices;
            m_VertexWeights = newWeights;
        }

        /// <summary>
        /// Sets the temporary vertex positions overrides.
        /// Overrides are not serialized.
        /// </summary>
        /// <param name="vertexPositionsOverride">Array of new vertex positions.</param>
        public void SetVertexPositionsOverride(Vector2[] vertexPositionsOverride)
        {
            if (vertexCount == vertexPositionsOverride.Length)
                m_VertexPositionsOverride = vertexPositionsOverride;
        }

        /// <summary>
        /// Clears the temporary vertex positions overrides.
        /// </summary>
        public void ClearVertexPositionOverride()
        {
            m_VertexPositionsOverride = null;
        }

        public void AddVertex(Vector2 position, BoneWeight weight)
        {
            ClearVertexPositionOverride();

            var listOfVertices = new List<Vector2>(m_Vertices);
            listOfVertices.Add(position);
            m_Vertices = listOfVertices.ToArray();

            var listOfWeights = new List<EditableBoneWeight>(m_VertexWeights);
            listOfWeights.Add(EditableBoneWeightUtility.CreateFromBoneWeight(weight));
            m_VertexWeights = listOfWeights.ToArray();
        }

        public void RemoveVertex(int index)
        {
            ClearVertexPositionOverride();

            var listOfVertices = new List<Vector2>(m_Vertices);
            listOfVertices.RemoveAt(index);
            m_Vertices = listOfVertices.ToArray();

            var listOfWeights = new List<EditableBoneWeight>(m_VertexWeights);
            listOfWeights.RemoveAt(index);
            m_VertexWeights = listOfWeights.ToArray();
        }

        public abstract SpriteBoneData GetBoneData(int index);

        public abstract float GetBoneDepth(int index);

        public void Clear()
        {
            m_Indices = new int[0];
            m_Vertices = new Vector2[0];
            m_VertexWeights = new EditableBoneWeight[0];
            m_Edges = new int2[0];
            m_OutlineEdges = new int2[0];
            m_VertexPositionsOverride = null;
        }
    }

    [Serializable]
    internal class SpriteMeshData : BaseSpriteMeshData
    {
        [SerializeField]
        List<SpriteBoneData> m_Bones = new List<SpriteBoneData>();

        [SerializeField]
        Rect m_Frame;

        public override Rect frame => m_Frame;
        public override int boneCount => m_Bones.Count;

        public List<SpriteBoneData> bones
        {
            get => m_Bones;
            set => m_Bones = value;
        }

        public override SpriteBoneData GetBoneData(int index)
        {
            return m_Bones[index];
        }

        public override float GetBoneDepth(int index)
        {
            return m_Bones[index].depth;
        }

        public void SetFrame(Rect newFrame)
        {
            m_Frame = newFrame;
        }
    }
}
