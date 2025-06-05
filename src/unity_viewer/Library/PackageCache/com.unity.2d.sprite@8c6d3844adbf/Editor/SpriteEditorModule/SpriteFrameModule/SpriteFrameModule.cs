using System;
using System.IO;
using UnityEngine;
using UnityEditorInternal;
using System.Collections.Generic;
using System.Text;
using UnityTexture2D = UnityEngine.Texture2D;
using UnityEditor.ShortcutManagement;
using Object = UnityEngine.Object;

namespace UnityEditor.U2D.Sprites
{
    [RequireSpriteDataProvider(typeof(ITextureDataProvider))]
    internal partial class SpriteFrameModule : SpriteFrameModuleBase
    {
        public enum AutoSlicingMethod
        {
            DeleteAll = 0,
            Smart = 1,
            Safe = 2
        }

        private bool[] m_AlphaPixelCache;
        SpriteFrameModuleContext m_SpriteFrameModuleContext;
        public event Action onModuleDeactivated = () => { };
        SpriteEditorModeBase m_CurrentMode = null;

        private StringBuilder m_SpriteNameStringBuilder;
        bool m_SpriteRectValidated = false;

        private List<Rect> m_PotentialRects;

        Texture2D m_TextureToSlice;
        public List<Rect> potentialRects
        {
            set => m_PotentialRects = value;
        }
        internal static Func<string, string, string, string, string, int> onShowComplexDialog = EditorUtility.DisplayDialogComplex;
        internal static Func<string, string, string, string, bool> onShowDialog = EditorUtility.DisplayDialog;
        public SpriteFrameModule(ISpriteEditor sw, IEventSystem es, IUndoSystem us, IAssetDatabase ad) :
            base("Sprite Editor", sw, es, us, ad)
        {}

        public override void SetModuleModes(IEnumerable<Type> modes)
        {
            base.SetModuleModes(modes);
            foreach (var mode in this.modes)
            {
                mode.RegisterOnModeRequestActivate(OnModuleExtensionActivate);
            }
        }

        void OnModuleExtensionActivate(SpriteEditorModeBase activatingMode)
        {
            m_CurrentMode?.DeactivateMode();
            m_CurrentMode = activatingMode;
            var activated = m_CurrentMode?.ActivateMode();
            // Mode did not activate
            if(activated.HasValue && !activated.Value)
            {
                m_CurrentMode = null;
            }

            bool modeNull = m_CurrentMode == null;
            if (modeNull)
            {
                spriteEditor.spriteRects = m_RectsCache.GetSpriteRects();
            }
            EnableInspector(modeNull);
        }

        public override bool ApplyRevert(bool apply)
        {
            var returnValue = base.ApplyRevert(apply);
            var dataProviderApplied = new HashSet<Type>();
            dataProviderApplied.Add(typeof(ISpriteEditorDataProvider));
            foreach(var mode in modes)
            {
                returnValue |= mode.ApplyModeData(apply, dataProviderApplied);
            }
            if(apply)
                m_SourceOverrideCallback?.Invoke(spriteAssetPath);
            return returnValue;
        }

        class SpriteFrameModuleContext : IShortcutContext
        {
            SpriteFrameModule m_SpriteFrameModule;

            public SpriteFrameModuleContext(SpriteFrameModule spriteFrame)
            {
                m_SpriteFrameModule = spriteFrame;
            }

            public bool active
            {
                get { return true; }
            }
            public SpriteFrameModule spriteFrameModule
            {
                get { return m_SpriteFrameModule; }
            }
        }

        [FormerlyPrefKeyAs("Sprite Editor/Trim", "#t")]
        [Shortcut("Sprite Editor/Trim", typeof(SpriteFrameModuleContext), KeyCode.T, ShortcutModifiers.Shift)]
        static void ShortcutTrim(ShortcutArguments args)
        {
            if (!string.IsNullOrEmpty(GUI.GetNameOfFocusedControl()))
                return;
            var spriteFrameContext = (SpriteFrameModuleContext)args.context;
            spriteFrameContext.spriteFrameModule.TrimAlpha();
            spriteFrameContext.spriteFrameModule.spriteEditor.RequestRepaint();
        }

        public override void OnModuleActivate()
        {
            base.OnModuleActivate();
            m_SpriteRectValidated = false;
            spriteEditor.enableMouseMoveEvent = true;
            m_SpriteFrameModuleContext = new SpriteFrameModuleContext(this);
            ShortcutIntegration.instance.contextManager.RegisterToolContext(m_SpriteFrameModuleContext);
            m_SpriteNameStringBuilder = new StringBuilder(GetSpriteNamePrefix() + "_");
            m_PotentialRects = null;
            RegisterDataChangeCallback(OnTextureDataProviderDataChanged);
            SignalModuleActivate();
        }

        void ValidateSpriteRects()
        {
            if (m_TextureDataProvider != null && !m_SpriteRectValidated)
            {
                m_SpriteRectValidated = true;
                int width, height;
                m_TextureDataProvider.GetTextureActualWidthAndHeight(out width, out height);
                HashSet<GUID> spriteIDs = new HashSet<GUID>();
                var emptyGuid = new GUID();
                List<SpriteRect> updatedSpriteRectID = new List<SpriteRect>();
                for (int i = 0; i < m_RectsCache.spriteRects.Count; ++i)
                {
                    // Validate rect is still within the bounds of the texture
                    var s = m_RectsCache.spriteRects[i];
                    if(s.rect.x < 0 || s.rect.y < 0 || s.rect.xMax > width || s.rect.yMax > height)
                    {
                        var response = onShowComplexDialog("Invalid Sprite Rect", $"Sprite Rect {s.name} is outside the bounds of the texture.", "Remove", "Keep", "Resize");
                        switch (response)
                        {
                            case 0:
                                m_RectsCache.Remove(s);
                                i--;
                                SetDataModified();
                                break;
                            case 2:
                                s.rect = ClampSpriteRect(s.rect, width, height);
                                SetDataModified();
                                break;
                        }
                    }

                    // Validate sprite id uniqueness
                    var spriteId = s.spriteID;
                    if (spriteId == emptyGuid || spriteIDs.Contains(spriteId))
                    {
                        if (onShowDialog("Invalid Sprite ID", $"Sprite Rect {s.name} has an invalid ID.\nSprite with invalid ID can result in Sprite reference breakage.", "Reassign ID", "Keep"))
                        {
                            updatedSpriteRectID.Add(s);
                        }
                    }
                    spriteIDs.Add(spriteId);
                }
                for(int i = 0; i < updatedSpriteRectID.Count; ++i)
                {
                    m_RectsCache.Remove(updatedSpriteRectID[i]);
                    updatedSpriteRectID[i].spriteID = GUID.Generate();
                    m_RectsCache.Add(updatedSpriteRectID[i], true);
                    SetDataModified();
                }
            }
        }

        public override void OnModuleDeactivate()
        {
            base.OnModuleDeactivate();
            EditorApplication.delayCall -= ValidateSpriteRects;
            m_SpriteRectValidated = true;
            ShortcutIntegration.instance.contextManager.DeregisterToolContext(m_SpriteFrameModuleContext);
            m_PotentialRects = null;
            m_AlphaPixelCache = null;
            m_CurrentMode?.DeactivateMode();
            m_CurrentMode = null;
            UnregisterDataChangeCallback(OnTextureDataProviderDataChanged);
            CleanUpDataDataProviderOverride();
            foreach (var mode in modes)
            {
                mode.UnregisterOnModeRequestActivate(OnModuleExtensionActivate);
            }
            if(m_TextureToSlice != null)
                Object.DestroyImmediate(m_TextureToSlice);
            modes.Clear();
            onModuleDeactivated();
        }

        void OnTextureDataProviderDataChanged(ISpriteEditorDataProvider obj)
        {
            if(m_TextureToSlice != null)
                Object.DestroyImmediate(m_TextureToSlice);
            m_TextureToSlice = null;
        }

        public static SpriteImportMode GetSpriteImportMode(ISpriteEditorDataProvider dataProvider)
        {
            return dataProvider == null ? SpriteImportMode.None : dataProvider.spriteImportMode;
        }

        public override bool CanBeActivated()
        {
            var mode = GetSpriteImportMode(spriteEditor.GetDataProvider<ISpriteEditorDataProvider>());
            return mode != SpriteImportMode.Polygon && mode != SpriteImportMode.None;
        }

        private string GenerateSpriteNameWithIndex(int startIndex)
        {
            int originalLength = m_SpriteNameStringBuilder.Length;
            m_SpriteNameStringBuilder.Append(startIndex);
            var name = m_SpriteNameStringBuilder.ToString();
            m_SpriteNameStringBuilder.Length = originalLength;
            return name;
        }

        private bool PixelHasAlpha(int x, int y, UnityTexture2D texture)
        {
            if (m_AlphaPixelCache == null)
            {
                m_AlphaPixelCache = new bool[texture.width * texture.height];
                Color32[] pixels = texture.GetPixels32();

                for (int i = 0; i < pixels.Length; i++)
                    m_AlphaPixelCache[i] = pixels[i].a != 0;
            }
            int index = y * (int)texture.width + x;
            return m_AlphaPixelCache[index];
        }

        private string GetSpriteNamePrefix()
        {
            return Path.GetFileNameWithoutExtension(spriteAssetPath);
        }

        public void DoAutomaticSlicing(int minimumSpriteSize, int alignment, Vector2 pivot, AutoSlicingMethod slicingMethod)
        {
            m_RectsCache.RegisterUndo(undoSystem, "Automatic Slicing");

            if (slicingMethod == AutoSlicingMethod.DeleteAll)
                m_RectsCache.Clear();

            var textureToUse = GetTextureToSlice();
            List<Rect> frames = new List<Rect>(InternalSpriteUtility.GenerateAutomaticSpriteRectangles(textureToUse, minimumSpriteSize, 0));
            if (frames.Count == 0)
                frames.Add(new Rect(0, 0, textureToUse.width, textureToUse.height));

            int index = 0;
            int originalCount = m_RectsCache.spriteRects.Count;

            foreach (Rect frame in frames)
                m_RectsCache.AddSprite(frame, alignment, pivot, slicingMethod, originalCount, ref index, GenerateSpriteNameWithIndex);

            if (slicingMethod == AutoSlicingMethod.DeleteAll)
                m_RectsCache.ClearUnusedFileID();
            selected = null;
            NotifyOnSpriteRectChanged();
            spriteEditor.SetDataModified();
            Repaint();
        }

        UnityTexture2D GetTextureToSlice()
        {
            m_TextureDataProvider.GetTextureActualWidthAndHeight(out var width, out var height);
            var readableTexture = m_TextureDataProvider.GetReadableTexture2D();
            if (readableTexture == null || (readableTexture.width == width && readableTexture.height == height))
                return readableTexture;
            // we want to slice based on the original texture slice. Upscale the imported texture
            if (m_TextureToSlice == null)
            {
                // we want to slice based on the original texture slice. Upscale the imported texture
                m_TextureToSlice = UnityEditor.SpriteUtility.CreateTemporaryDuplicate(readableTexture, width, height);
                m_TextureToSlice.hideFlags = HideFlags.HideAndDontSave;
            }

            return m_TextureToSlice;
        }

        public IEnumerable<Rect> GetGridRects(Vector2 size, Vector2 offset, Vector2 padding, bool keepEmptyRects)
        {
            var textureToUse = GetTextureToSlice();
            return InternalSpriteUtility.GenerateGridSpriteRectangles(textureToUse, offset, size, padding, keepEmptyRects);
        }

        public void DoGridSlicing(Vector2 size, Vector2 offset, Vector2 padding, int alignment, Vector2 pivot, AutoSlicingMethod slicingMethod, bool keepEmptyRects = false)
        {
            var frames = GetGridRects(size, offset, padding, keepEmptyRects);

            m_RectsCache.RegisterUndo(undoSystem, "Grid Slicing");
            if (slicingMethod == AutoSlicingMethod.DeleteAll)
                m_RectsCache.Clear();

            int index = 0;
            int originalCount = m_RectsCache.spriteRects.Count;
            foreach (Rect frame in frames)
                m_RectsCache.AddSprite(frame, alignment, pivot, slicingMethod, originalCount, ref index, GenerateSpriteNameWithIndex);

            if (slicingMethod == AutoSlicingMethod.DeleteAll)
                m_RectsCache.ClearUnusedFileID();
            selected = null;
            NotifyOnSpriteRectChanged();
            spriteEditor.SetDataModified();
            Repaint();
        }

        public void DoIsometricGridSlicing(Vector2 size, Vector2 offset, int alignment, Vector2 pivot, AutoSlicingMethod slicingMethod, bool keepEmptyRects = false, bool isAlternate = false)
        {
            var frames = IsometricSlicingUtility.GetIsometricRects(GetTextureToSlice(), size, offset, isAlternate, keepEmptyRects);

            List<Vector2[]> outlines = new List<Vector2[]>(4);
            outlines.Add(new[] { new Vector2(0.0f, -size.y / 2)
                                 , new Vector2(size.x / 2, 0.0f)
                                 , new Vector2(0.0f, size.y / 2)
                                 , new Vector2(-size.x / 2, 0.0f)});

            m_RectsCache.RegisterUndo(undoSystem, "Isometric Grid Slicing");
            if (slicingMethod == AutoSlicingMethod.DeleteAll)
                m_RectsCache.Clear();

            int index = 0;
            var spriteRects = m_RectsCache.GetSpriteRects();
            int originalCount = spriteRects.Count;
            foreach (var frame in frames)
            {
                var spriteIndex = m_RectsCache.AddSprite(frame, alignment, pivot, slicingMethod, originalCount, ref index, GenerateSpriteNameWithIndex);
                var outlineRect = new OutlineSpriteRect(spriteRects[spriteIndex]);
                outlineRect.outlines = outlines;
                spriteRects[spriteIndex] = outlineRect;
            }
            if (slicingMethod == AutoSlicingMethod.DeleteAll)
                m_RectsCache.ClearUnusedFileID();
            selected = null;
            NotifyOnSpriteRectChanged();
            spriteEditor.SetDataModified();
            Repaint();
        }

        public void ScaleSpriteRect(Rect r)
        {
            if (selected != null)
            {
                m_RectsCache.RegisterUndo(undoSystem, "Scale sprite");
                selected.rect = ClampSpriteRect(r, textureActualWidth, textureActualHeight);
                selected.border = ClampSpriteBorderToRect(selected.border, selected.rect);
                NotifyOnSpriteRectChanged();
                spriteEditor.SetDataModified();
            }
        }

        public void TrimAlpha()
        {
            var texture = GetTextureToSlice();
            if (texture == null)
                return;

            Rect rect = selected.rect;

            int xMin = (int)rect.xMax;
            int xMax = (int)rect.xMin;
            int yMin = (int)rect.yMax;
            int yMax = (int)rect.yMin;

            for (int y = (int)rect.yMin; y < (int)rect.yMax; y++)
            {
                for (int x = (int)rect.xMin; x < (int)rect.xMax; x++)
                {
                    if (PixelHasAlpha(x, y, texture))
                    {
                        xMin = Mathf.Min(xMin, x);
                        xMax = Mathf.Max(xMax, x);
                        yMin = Mathf.Min(yMin, y);
                        yMax = Mathf.Max(yMax, y);
                    }
                }
            }
            // Case 582309: Return an empty rectangle if no pixel has an alpha
            if (xMin > xMax || yMin > yMax)
                rect = new Rect(0, 0, 0, 0);
            else
                rect = new Rect(xMin, yMin, xMax - xMin + 1, yMax - yMin + 1);

            if (rect.width <= 0 && rect.height <= 0)
            {
                m_RectsCache.Remove(selected);
                spriteEditor.SetDataModified();
                selected = null;
            }
            else
            {
                rect = ClampSpriteRect(rect, texture.width, texture.height);
                if (selected.rect != rect)
                    spriteEditor.SetDataModified();

                selected.rect = rect;
                PopulateSpriteFrameInspectorField();
            }
        }

        public void DuplicateSprite()
        {
            if (selected != null)
            {
                m_RectsCache.RegisterUndo(undoSystem, "Duplicate sprite");
                var index = 0;
                var createdIndex = -1;
                while (createdIndex == -1)
                {
                    createdIndex = m_RectsCache.AddSprite(selected.rect, (int)selected.alignment, selected.pivot, GenerateSpriteNameWithIndex(index++), selected.border);
                }
                spriteEditor.SetDataModified();
                selected = m_RectsCache.spriteRects[createdIndex];
                NotifyOnSpriteRectChanged();
            }
        }

        public void CreateSprite(Rect rect)
        {
            rect = ClampSpriteRect(rect, textureActualWidth, textureActualHeight);
            m_RectsCache.RegisterUndo(undoSystem, "Create sprite");
            var index = 0;
            var createdIndex = -1;
            while (createdIndex == -1)
            {
                createdIndex = m_RectsCache.AddSprite(rect, 0, Vector2.zero, GenerateSpriteNameWithIndex(index++), Vector4.zero);
            }
            spriteEditor.SetDataModified();
            selected = m_RectsCache.spriteRects[createdIndex];
            NotifyOnSpriteRectChanged();
        }

        public void DeleteSprite()
        {
            if (selected != null)
            {
                m_RectsCache.RegisterUndo(undoSystem, "Delete sprite");
                m_RectsCache.Remove(selected);
                selected = null;
                NotifyOnSpriteRectChanged();
                spriteEditor.SetDataModified();
            }
        }

        public bool IsOnlyUsingDefaultNamedSpriteRects()
        {
            var onlyDefaultNames = true;
            var names = m_RectsCache.spriteNames;
            var defaultName = m_SpriteNameStringBuilder.ToString();

            foreach (var name in names)
            {
                if (!name.Contains(defaultName))
                {
                    onlyDefaultNames = false;
                    break;
                }
            }

            return onlyDefaultNames;
        }
    }
}
