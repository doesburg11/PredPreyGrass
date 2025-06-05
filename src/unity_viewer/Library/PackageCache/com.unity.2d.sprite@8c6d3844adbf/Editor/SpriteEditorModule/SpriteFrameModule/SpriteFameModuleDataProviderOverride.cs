using System;
using System.Collections.Generic;
using UnityEngine;
using UnityObject = UnityEngine.Object;

namespace UnityEditor.U2D.Sprites
{
    internal partial class SpriteFrameModule : SpriteFrameModuleBase, ISpriteEditorDataProvider, ITextureDataProvider
    {
        void CleanUpDataDataProviderOverride()
        {
            m_OriginalSpriteDataProvider = null;
            m_OriginalTextureDataProvider = null;
            m_SourceOverrideCallback = null;
            m_MainTexture = null;
            m_PreviewTexture = null;
            m_ReadableTexture = null;
        }

        #region ISpriteEditor implementation override for SpriteEditorMode support
        public override void SetPreviewTexture(Texture2D texture, int width, int height)
        {
            var targetObject = spriteEditor?.GetDataProvider<ISpriteEditorDataProvider>()?.targetObject;
            if(targetObject != null && targetObject ==  m_OriginalSpriteDataProvider?.targetObject)
                base.SetPreviewTexture(texture, width, height);
        }
        #endregion

        #region ISpriteEditorDataProvider implementation
        ISpriteEditorDataProvider m_OriginalSpriteDataProvider;

        ISpriteEditorDataProvider originalSpriteEditorDataProvider
        {
            get
            {
                if (m_OriginalSpriteDataProvider == null)
                {
                    m_OriginalSpriteDataProvider = spriteEditor.GetDataProvider<ISpriteEditorDataProvider>();
                    m_OriginalSpriteDataProvider.InitSpriteEditorDataProvider();
                }
                return m_OriginalSpriteDataProvider;
            }
        }
        event Action<ISpriteEditorDataProvider> m_OnSpriteEditorDataChangeCallback = _ => { };
        SpriteImportMode ISpriteEditorDataProvider.spriteImportMode => originalSpriteEditorDataProvider.spriteImportMode;
        public float pixelsPerUnit => originalSpriteEditorDataProvider.pixelsPerUnit;
        public UnityObject targetObject => originalSpriteEditorDataProvider.targetObject;

        public SpriteRect[] GetSpriteRects()
        {
            if (m_RectsCache == null)
            {
                if (originalSpriteEditorDataProvider.targetObject != null)
                    return originalSpriteEditorDataProvider.GetSpriteRects();
                return Array.Empty<SpriteRect>();
            }

            return m_RectsCache.GetSpriteRects().ToArray();
        }

        public void SetSpriteRects(SpriteRect[] spriteRects)
        {
            m_RectsCache.SetSpriteRects(spriteRects);
            spriteEditor.spriteRects = new List<SpriteRect>(spriteRects);
            NotifyOnSpriteRectChanged();
        }

        public void Apply()
        {
            originalSpriteEditorDataProvider.Apply();
        }

        public void InitSpriteEditorDataProvider()
        { }

        public override T GetDataProvider<T>() where T : class
        {
            if(typeof(T) == typeof(ISpriteEditorDataProvider))
                return this as T;
            else if (typeof(T) == typeof(ITextureDataProvider))
                return this as T;
            return spriteEditor.GetDataProvider<T>();
        }

        public bool HasDataProvider(Type type)
        {
            return originalSpriteEditorDataProvider.HasDataProvider(type);
        }

        public void RegisterDataChangeCallback(Action<ISpriteEditorDataProvider> action)
        {
            m_OnSpriteEditorDataChangeCallback+= action;
        }

        public void UnregisterDataChangeCallback(Action<ISpriteEditorDataProvider> action)
        {
            m_OnSpriteEditorDataChangeCallback-= action;
        }

        protected override void NotifyOnSpriteRectChanged()
        {
            PopulateSpriteFrameInspectorField();
            m_OnSpriteEditorDataChangeCallback.Invoke(this);
        }
        #endregion

        #region ITextureDataProvider implementation

        ITextureDataProvider m_OriginalTextureDataProvider;

        ITextureDataProvider originalTextureDataProvider
        {
            get
            {
                if (m_OriginalTextureDataProvider == null)
                {
                    m_OriginalTextureDataProvider = spriteEditor.GetDataProvider<ITextureDataProvider>();
                }
                return m_OriginalTextureDataProvider;
            }
        }

        Texture2D m_MainTexture;
        Texture2D m_PreviewTexture;
        Texture2D m_ReadableTexture;
        int m_OverrideSourceTextureWidth;
        int m_OverrideSourceTextureHeight;
        event Action<ITextureDataProvider> m_OnTextureDataChangeCallback = _ => { };
        Action<string> m_SourceOverrideCallback;
        public Texture2D texture => m_MainTexture ?? originalTextureDataProvider.texture;
        public Texture2D previewTexture => m_PreviewTexture ?? originalTextureDataProvider.previewTexture;
        public void GetTextureActualWidthAndHeight(out int width, out int height)
        {
            if(m_MainTexture != null)
            {
                width = m_OverrideSourceTextureWidth;
                height = m_OverrideSourceTextureHeight;
            }
            else
                originalTextureDataProvider.GetTextureActualWidthAndHeight(out width, out height);
        }

        public Texture2D GetReadableTexture2D()
        {
            if (m_MainTexture != null && m_ReadableTexture == null)
            {
                m_ReadableTexture = UnityEditor.SpriteUtility.CreateTemporaryDuplicate(m_MainTexture, m_MainTexture.width, m_MainTexture.height);
                if (m_ReadableTexture != null)
                    m_ReadableTexture.filterMode = m_MainTexture.filterMode;
            }
            return m_ReadableTexture ?? originalTextureDataProvider?.GetReadableTexture2D();
        }

        public bool OverrideTextures(Texture2D mainTexture, Texture2D previewTexture, int width, int height)
        {
            m_MainTexture = mainTexture;
            m_PreviewTexture = previewTexture;
            m_ReadableTexture = null;
            m_OverrideSourceTextureWidth = width;
            m_OverrideSourceTextureHeight = height;
            m_OnTextureDataChangeCallback.Invoke(this);
            return true;
        }

        /// <summary>
        /// Registers a callback to override the source texture.
        /// </summary>
        /// <param name="action">Callback that will write to the source texture with the path of the source texture.</param>
        public void RegisterSourceTextureOverride(Action<string> action)
        {
            m_SourceOverrideCallback = action;
        }

        public void UnregisterSourceTextureOverride(Action<string> action)
        {
            if (m_SourceOverrideCallback == action)
                m_SourceOverrideCallback = null;
        }

        public void RegisterDataChangeCallback(Action<ITextureDataProvider> action)
        {
            m_OnTextureDataChangeCallback += action;
        }

        public void UnregisterDataChangeCallback(Action<ITextureDataProvider> action)
        {
            m_OnTextureDataChangeCallback -= action;
        }
        #endregion
    }
}
