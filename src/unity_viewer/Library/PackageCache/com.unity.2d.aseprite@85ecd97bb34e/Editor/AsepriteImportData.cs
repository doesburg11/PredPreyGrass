using UnityEngine;

namespace UnityEditor.U2D.Aseprite
{
    internal class AsepriteImportData : ScriptableObject
    {
        [SerializeField]
        int m_ImportedTextureWidth;
        public int importedTextureWidth
        {
            get => m_ImportedTextureWidth;
            set => m_ImportedTextureWidth = value;
        }

        [SerializeField]
        int m_ImportedTextureHeight;
        public int importedTextureHeight
        {
            get => m_ImportedTextureHeight;
            set => m_ImportedTextureHeight = value;
        }

        [SerializeField]
        int m_TextureActualHeight;
        public int textureActualHeight
        {
            get => m_TextureActualHeight;
            set => m_TextureActualHeight = value;
        }

        [SerializeField]
        int m_TextureActualWidth;
        public int textureActualWidth
        {
            get => m_TextureActualWidth;
            set => m_TextureActualWidth = value;
        }
    }
}
