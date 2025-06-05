using UnityEditor.U2D;
using UnityEditorInternal;
using UnityEngine;
using UnityEngine.U2D;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    /// <summary>
    /// Editor for a TileSetImporter
    /// </summary>
    [CustomEditor(typeof(TileSetImporter))]
    public class TileSetImporterEditor : Editor
    {
        private string m_AssetPath;
        private TileSet m_TileSet;
        private SpriteAtlas m_SpriteAtlas;

        private SerializedObject m_SerializedTileSetObject;
        private SerializedObject m_SerializedSpriteAtlasObject;

        private TileSetImporterElement m_MainElement;

        private Editor m_SpriteAtlasEditor;

        private SerializedObject serializedTileSetObject
        {
            get
            {
                if (m_SerializedTileSetObject == null)
                {
                    try
                    {
                        m_SerializedTileSetObject = new SerializedObject(m_TileSet);
                    }
                    catch (System.ArgumentException e)
                    {
                        m_SerializedTileSetObject = null;
                        throw e;
                    }
                }
                return m_SerializedTileSetObject;
            }
        }

        private SerializedObject serializedSpriteAtlasObject
        {
            get
            {
                if (m_SerializedSpriteAtlasObject == null)
                {
                    try
                    {
                        m_SerializedSpriteAtlasObject = new SerializedObject(m_SpriteAtlas);
                    }
                    catch (System.ArgumentException e)
                    {
                        m_SerializedSpriteAtlasObject = null;
                        throw e;
                    }
                }
                return m_SerializedSpriteAtlasObject;
            }
        }

        private void OnEnable()
        {
            LoadTileSet();
        }

        private void LoadTileSet()
        {
            m_AssetPath = AssetDatabase.GetAssetPath(target);
            var loadedObjects = InternalEditorUtility.LoadSerializedFileAndForget(m_AssetPath);
            foreach (var loadedObject in loadedObjects)
            {
                if (loadedObject is TileSet tileSet)
                    m_TileSet = tileSet;
                if (loadedObject is SpriteAtlas spriteAtlas)
                    m_SpriteAtlas = spriteAtlas;
            }
            if (m_SpriteAtlas == null)
            {
                m_SpriteAtlas = new SpriteAtlas();
                var sats = new SpriteAtlasTextureSettings()
                {
                    filterMode = FilterMode.Point,
                    anisoLevel = 0,
                    generateMipMaps = false,
                    sRGB = true,
                };
                m_SpriteAtlas.SetTextureSettings(sats);
            }
        }

        /// <summary>
        /// Creates VisualElements for the Inspector GUI of a TileSetImporter.
        /// </summary>
        /// <returns>
        /// Root VisualElement for the Inspector GUI of a TileSetImporter
        /// </returns>
        public override VisualElement CreateInspectorGUI()
        {
            m_MainElement = new TileSetImporterElement(serializedObject)
            {
                onRevert = Revert,
                onApply = ApplyAndImport
            };
            m_MainElement.Bind(serializedTileSetObject, m_SpriteAtlas);
            return m_MainElement;
        }

        private void Revert()
        {
            m_SerializedTileSetObject = null;
            m_SerializedSpriteAtlasObject = null;
            LoadTileSet();
            if (m_MainElement != null)
                m_MainElement.Bind(serializedTileSetObject, m_SpriteAtlas);
        }

        private void ApplyAndImport()
        {
            serializedObject.ApplyModifiedProperties();
            serializedTileSetObject.ApplyModifiedPropertiesWithoutUndo();

            foreach (var textureSource in m_TileSet.textureSources)
            {
                m_SpriteAtlas.Remove(new Object[] { textureSource.texture });
                m_SpriteAtlas.Add(new Object[] { textureSource.texture });
            }

            InternalEditorUtility.SaveToSerializedFileAndForget(new Object[] {
                m_TileSet, m_SpriteAtlas
            }, m_AssetPath, EditorSettings.serializationMode != SerializationMode.ForceBinary);
            AssetDatabase.ImportAsset(m_AssetPath);
        }
    }
}
