using System;
using System.Collections.Generic;
using UnityEditor.U2D.Aseprite.Common;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEditor.U2D.Sprites;
using UnityEngine.U2D;

namespace UnityEditor.U2D.Aseprite
{
    internal abstract class AsepriteDataProvider
    {
        public AsepriteImporter dataProvider;
    }

    internal class SpriteBoneDataProvider : AsepriteDataProvider, ISpriteBoneDataProvider
    {
        public List<SpriteBone> GetBones(GUID guid)
        {
            var sprite = ((SpriteMetaData)dataProvider.GetSpriteData(guid));
            Assert.IsNotNull(sprite, $"Sprite not found for GUID:{guid.ToString()}");
            var returnValue = new List<SpriteBone>();
            if (sprite.spriteBone != null)
            {
                returnValue.AddRange(sprite.spriteBone);
            }

            return returnValue;
        }

        public void SetBones(GUID guid, List<SpriteBone> bones)
        {
            var sprite = dataProvider.GetSpriteData(guid);
            if (sprite != null)
                ((SpriteMetaData)sprite).spriteBone = bones;
        }
    }

    internal class SpriteFrameEditCapabilityDataProvider : AsepriteDataProvider, ISpriteFrameEditCapability
    {
        static EditCapability k_DefaultEditCapability = new EditCapability(EEditCapability.EditPivot);
        static EditCapability k_SpriteSheetCapabilities = new EditCapability(EEditCapability.All);

    public EditCapability GetEditCapability()
        {
            switch (dataProvider.importMode)
            {
                case FileImportModes.SpriteSheet:
                    return k_SpriteSheetCapabilities;
                default:
                    return k_DefaultEditCapability;
            }
        }

        public void SetEditCapability(EditCapability editCapability)
        {

        }
    }

    internal class TextureDataProvider : AsepriteDataProvider, ITextureDataProvider
    {
        Texture2D m_ReadableTexture;
        Texture2D m_OriginalTexture;

        AsepriteImporter textureImporter => (AsepriteImporter)dataProvider.targetObject;

        public Texture2D texture
        {
            get
            {
                if (m_OriginalTexture == null)
                    m_OriginalTexture = AssetDatabase.LoadAssetAtPath<Texture2D>(textureImporter.assetPath);
                return m_OriginalTexture;
            }
        }

        public Texture2D previewTexture => texture;

        public Texture2D GetReadableTexture2D()
        {
            if (m_ReadableTexture == null)
            {
                m_ReadableTexture = InternalEditorBridge.CreateTemporaryDuplicate(texture, texture.width, texture.height);
                if (m_ReadableTexture != null)
                    m_ReadableTexture.filterMode = texture.filterMode;
            }
            return m_ReadableTexture;
        }

        public void GetTextureActualWidthAndHeight(out int width, out int height)
        {
            width = dataProvider.textureActualWidth;
            height = dataProvider.textureActualHeight;
        }
    }

    internal class SecondaryTextureDataProvider : AsepriteDataProvider, ISecondaryTextureDataProvider
    {
        public SecondarySpriteTexture[] textures
        {
            get => dataProvider.secondaryTextures;
            set => dataProvider.secondaryTextures = value;
        }
    }

    internal class SpriteOutlineDataProvider : AsepriteDataProvider, ISpriteOutlineDataProvider
    {
        public List<Vector2[]> GetOutlines(GUID guid)
        {
            var sprite = ((SpriteMetaData)dataProvider.GetSpriteData(guid));
            Assert.IsNotNull(sprite, $"Sprite not found for GUID:{guid.ToString()}");

            var outline = sprite.spriteOutline;
            var returnValue = new List<Vector2[]>();
            if (outline != null)
            {
                for (int i = 0; i < outline.Count; ++i)
                {
                    returnValue.Add(outline[i].outline);
                }
            }

            return returnValue;
        }

        public void SetOutlines(GUID guid, List<Vector2[]> data)
        {
            var sprite = dataProvider.GetSpriteData(guid);
            if (sprite == null)
                return;

            var list = new List<SpriteOutline>();
            foreach (var outline in data)
            {
                list.Add(new SpriteOutline() { outline = outline });
            }
            ((SpriteMetaData)sprite).spriteOutline = list;
        }

        public float GetTessellationDetail(GUID guid)
        {
            return ((SpriteMetaData)dataProvider.GetSpriteData(guid)).tessellationDetail;
        }

        public void SetTessellationDetail(GUID guid, float value)
        {
            var sprite = dataProvider.GetSpriteData(guid);
            if (sprite != null)
                ((SpriteMetaData)sprite).tessellationDetail = value;
        }
    }

    internal class SpritePhysicsOutlineProvider : AsepriteDataProvider, ISpritePhysicsOutlineDataProvider
    {
        public List<Vector2[]> GetOutlines(GUID guid)
        {
            var sprite = ((SpriteMetaData)dataProvider.GetSpriteData(guid));
            Assert.IsNotNull(sprite, string.Format("Sprite not found for GUID:{0}", guid.ToString()));
            var outline = sprite.spritePhysicsOutline;
            var returnValue = new List<Vector2[]>();
            if (outline == null)
                return returnValue;

            foreach (var spriteOutline in outline)
                returnValue.Add(spriteOutline.outline);
            return returnValue;
        }

        public void SetOutlines(GUID guid, List<Vector2[]> data)
        {
            var sprite = dataProvider.GetSpriteData(guid);
            if (sprite == null)
                return;

            var list = new List<SpriteOutline>();
            foreach (var outline in data)
            {
                list.Add(new SpriteOutline() { outline = outline });
            }
            ((SpriteMetaData)sprite).spritePhysicsOutline = list;
        }

        public float GetTessellationDetail(GUID guid)
        {
            return ((SpriteMetaData)dataProvider.GetSpriteData(guid)).tessellationDetail;
        }

        public void SetTessellationDetail(GUID guid, float value)
        {
            var sprite = dataProvider.GetSpriteData(guid);
            if (sprite != null)
                ((SpriteMetaData)sprite).tessellationDetail = value;
        }
    }

    internal class SpriteMeshDataProvider : AsepriteDataProvider, ISpriteMeshDataProvider
    {
        public Vertex2DMetaData[] GetVertices(GUID guid)
        {
            var sprite = ((SpriteMetaData)dataProvider.GetSpriteData(guid));
            Assert.IsNotNull(sprite, $"Sprite not found for GUID:{guid.ToString()}");
            var v = sprite.vertices;
            if (v == null)
                return Array.Empty<Vertex2DMetaData>();

            var returnValue = new Vertex2DMetaData[v.Count];
            for (var i = 0; i < returnValue.Length; ++i)
            {
                returnValue[i] = v[i];
            }
            return returnValue;
        }

        public void SetVertices(GUID guid, Vertex2DMetaData[] vertices)
        {
            var sprite = dataProvider.GetSpriteData(guid);
            if (sprite != null)
                ((SpriteMetaData)sprite).vertices = new List<Vertex2DMetaData>(vertices);
        }

        public int[] GetIndices(GUID guid)
        {
            var sprite = ((SpriteMetaData)dataProvider.GetSpriteData(guid));
            Assert.IsNotNull(sprite, $"Sprite not found for GUID:{guid.ToString()}");
            var v = sprite.indices;
            if (v != null)
                return v;

            return Array.Empty<int>();
        }

        public void SetIndices(GUID guid, int[] indices)
        {
            var sprite = dataProvider.GetSpriteData(guid);
            if (sprite != null)
                ((SpriteMetaData)sprite).indices = indices;
        }

        public Vector2Int[] GetEdges(GUID guid)
        {
            var sprite = ((SpriteMetaData)dataProvider.GetSpriteData(guid));
            Assert.IsNotNull(sprite, $"Sprite not found for GUID:{guid.ToString()}");
            var v = sprite.edges;
            if (v != null)
                return v;

            return Array.Empty<Vector2Int>();
        }

        public void SetEdges(GUID guid, Vector2Int[] edges)
        {
            var sprite = dataProvider.GetSpriteData(guid);
            if (sprite != null)
                ((SpriteMetaData)sprite).edges = edges;
        }
    }
}