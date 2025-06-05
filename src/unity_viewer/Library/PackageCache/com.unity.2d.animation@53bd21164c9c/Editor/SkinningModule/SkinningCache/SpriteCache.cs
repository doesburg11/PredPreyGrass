using UnityEngine;

namespace UnityEditor.U2D.Animation
{
    internal class SpriteCache : TransformCache
    {
        [SerializeField]
        string m_ID;
        [SerializeField]
        Rect m_TextureRect;
        [SerializeField]
        Vector2 m_PivotNormalized;

        public string id
        {
            get => m_ID;
            internal set => m_ID = value;
        }

        public Rect textureRect
        {
            get => m_TextureRect;
            set => m_TextureRect = value;
        }

        public Vector2 pivotRectSpace => Vector2.Scale(textureRect.size, m_PivotNormalized);
    }
}
