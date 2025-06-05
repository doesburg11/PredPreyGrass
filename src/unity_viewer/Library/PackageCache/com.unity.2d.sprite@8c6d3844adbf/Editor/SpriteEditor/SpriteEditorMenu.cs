using UnityEngine;
using System;
using System.Collections.Generic;
using UnityEngine.Experimental.Rendering;
using UnityEvent = UnityEngine.Event;

namespace UnityEditor.U2D.Sprites
{
    [Serializable]
    internal class SpriteEditorMenuSettingObject : ScriptableObject
    {
        [SerializeField]
        public SpriteEditorMenuSetting settings;
    }

    [Serializable]
    internal class SpriteEditorMenuSetting
    {
        public const string kSliceOnImportKey = "SpriteEditor.SliceOnImport";
        public const string kSliceSettingsKey = "SpriteEditor.SliceSettings";

        public enum SlicingType
        {
            Automatic = 0,
            GridByCellSize = 1,
            GridByCellCount = 2,
            IsometricGrid = 3
        }

        [SerializeField]
        public bool sliceOnImport = false;
        [SerializeField]
        public Vector2 gridCellCount = new Vector2(1, 1);
        [SerializeField]
        public Vector2 gridSpriteSize = new Vector2(64, 64);
        [SerializeField]
        public Vector2 gridSpriteOffset = new Vector2(0, 0);
        [SerializeField]
        public Vector2 gridSpritePadding = new Vector2(0, 0);
        [SerializeField]
        public Vector2 pivot = Vector2.zero;
        [SerializeField]
        public int autoSlicingMethod = (int)SpriteFrameModule.AutoSlicingMethod.DeleteAll;
        [SerializeField]
        public int spriteAlignment;
        [SerializeField]
        public SlicingType slicingType;
        [SerializeField]
        public bool keepEmptyRects;
        [SerializeField]
        public bool isAlternate;
    }

    internal class SpriteEditorMenu : EditorWindow
    {
        private static Styles s_Styles;
        private static long s_LastClosedTime;
        private static SpriteEditorMenuSettingObject s_SettingsObject;
        private ITextureDataProvider m_TextureDataProvider;
        private ISpriteCustomDataProvider m_CustomDataProvider;
        private bool m_CanSliceOnImport;
        private SpriteFrameModule m_SpriteFrameModule;
        private List<Rect> m_PotentialRects;

        private static SpriteEditorMenuSetting s_Setting => s_SettingsObject.settings;

        private class Styles
        {
            public GUIStyle background = "grey_border";
            public GUIStyle notice;

            public Styles()
            {
                notice = new GUIStyle(GUI.skin.label);
                notice.alignment = TextAnchor.MiddleCenter;
                notice.wordWrap = true;
            }

            public readonly GUIContent[] spriteAlignmentOptions =
            {
                EditorGUIUtility.TrTextContent("Center"),
                EditorGUIUtility.TrTextContent("Top Left"),
                EditorGUIUtility.TrTextContent("Top"),
                EditorGUIUtility.TrTextContent("Top Right"),
                EditorGUIUtility.TrTextContent("Left"),
                EditorGUIUtility.TrTextContent("Right"),
                EditorGUIUtility.TrTextContent("Bottom Left"),
                EditorGUIUtility.TrTextContent("Bottom"),
                EditorGUIUtility.TrTextContent("Bottom Right"),
                EditorGUIUtility.TrTextContent("Custom")
            };

            public readonly GUIContent[] slicingMethodOptions =
            {
                EditorGUIUtility.TrTextContent("Delete Existing"),
                EditorGUIUtility.TrTextContent("Smart"),
                EditorGUIUtility.TrTextContent("Safe")
            };
            public readonly string[] slicingMethodInfoText =
            {
                L10n.Tr("Delete Existing removes all existing Sprites and recreates them from scratch."),
                L10n.Tr("Smart attempts to create new Sprites while retaining or adjusting existing ones. This slicing method does not remove any existing Sprites."),
                L10n.Tr("Safe adds new Sprites without changing anything already in place. This slicing method does not remove any existing Sprites.")
            };

            public readonly GUIContent methodLabel = EditorGUIUtility.TrTextContent("Method");
            public readonly GUIContent pivotLabel = EditorGUIUtility.TrTextContent("Pivot");
            public readonly GUIContent typeLabel = EditorGUIUtility.TrTextContent("Type");
            public readonly GUIContent sliceOnImportLabel = EditorGUIUtility.TrTextContent("Slice on Import", "Re-slices sprites from Texture when Texture is imported using slice settings if sliced before");
            public readonly GUIContent sliceButtonLabel = EditorGUIUtility.TrTextContent("Slice");
            public readonly GUIContent columnAndRowLabel = EditorGUIUtility.TrTextContent("Column & Row");
            public readonly GUIContent columnLabel = EditorGUIUtility.TextContent("C");
            public readonly GUIContent rowLabel = EditorGUIUtility.TextContent("R");
            public readonly GUIContent pixelSizeLabel = EditorGUIUtility.TrTextContent("Pixel Size");
            public readonly GUIContent xLabel = EditorGUIUtility.TextContent("X");
            public readonly GUIContent yLabel = EditorGUIUtility.TextContent("Y");
            public readonly GUIContent offsetLabel = EditorGUIUtility.TrTextContent("Offset");
            public readonly GUIContent paddingLabel = EditorGUIUtility.TrTextContent("Padding");
            public readonly GUIContent automaticSlicingHintLabel = EditorGUIUtility.TrTextContent("Texture is in compressed format. For better result please use manual slicing.", "Compressed textures may have artifacts that will affect the automatic slicing result. It is recommended to use manual slicing for better result.");
            public readonly GUIContent customPivotLabel = EditorGUIUtility.TrTextContent("Custom Pivot");
            public readonly GUIContent keepEmptyRectsLabel = EditorGUIUtility.TrTextContent("Keep Empty Rects");
            public readonly GUIContent isAlternateLabel = EditorGUIUtility.TrTextContent("Is Alternate");

            public readonly string deleteExistingTitle = L10n.Tr("Potential loss of Sprite data");
            public readonly string deleteExistingMessage = L10n.Tr("The Delete Existing slicing method recreates all Sprites with their default names. Renamed Sprites will lose their data in the process, and references to these Sprites will be lost. \n\nDo you wish you continue?");
            public readonly string yes = L10n.Tr("Yes");
            public readonly string no = L10n.Tr("No");
        }

        internal List<Rect> GetPotentialRects()
        {
            if (m_PotentialRects == null)
                m_PotentialRects = new List<Rect>();
            m_PotentialRects.Clear();
            switch (s_Setting.slicingType)
            {
                case SpriteEditorMenuSetting.SlicingType.Automatic:
                    // Do not show rects for Automatic
                    break;
                case SpriteEditorMenuSetting.SlicingType.GridByCellCount:
                    DetermineGridCellSizeWithCellCount(out var cellSize);
                    m_PotentialRects.AddRange(m_SpriteFrameModule.GetGridRects(cellSize
                        , s_Setting.gridSpriteOffset
                        , s_Setting.gridSpritePadding
                        , true));
                    break;
                case SpriteEditorMenuSetting.SlicingType.GridByCellSize:
                    m_PotentialRects.AddRange(m_SpriteFrameModule.GetGridRects(s_Setting.gridSpriteSize
                        , s_Setting.gridSpriteOffset
                        , s_Setting.gridSpritePadding
                        , true));
                    break;
                case SpriteEditorMenuSetting.SlicingType.IsometricGrid:
                    var texture = m_TextureDataProvider.GetReadableTexture2D();
                    if (texture != null)
                        m_PotentialRects.AddRange(IsometricSlicingUtility.GetIsometricRects(texture
                            , s_Setting.gridSpriteSize
                            , s_Setting.gridSpriteOffset
                            , s_Setting.isAlternate
                            , true));
                    break;
            }

            return m_PotentialRects;
        }

        private void Init(Rect buttonRect
            , SpriteFrameModule sf
            , ITextureDataProvider dataProvider
            , ISpriteCustomDataProvider customDataProvider
            , ISpriteFrameEditCapability capability)
        {
            // Create for once if setting was not created before.
            if (s_SettingsObject == null)
            {
                s_SettingsObject = CreateInstance<SpriteEditorMenuSettingObject>();
                s_SettingsObject.settings = new SpriteEditorMenuSetting();
            }

            m_SpriteFrameModule = sf;
            m_SpriteFrameModule.onModuleDeactivated -= OnModuleDeactivate;
            m_SpriteFrameModule.onModuleDeactivated += OnModuleDeactivate;
            m_TextureDataProvider = dataProvider;
            m_CustomDataProvider = customDataProvider;

            s_Setting.sliceOnImport = false;
            if (m_CustomDataProvider != null)
            {
                m_CanSliceOnImport = capability != null && capability.GetEditCapability().HasCapability(EEditCapability.SliceOnImport);
                m_CustomDataProvider.GetData(SpriteEditorMenuSetting.kSliceOnImportKey, out var sliceOnImportData);
                if (!String.IsNullOrEmpty(sliceOnImportData))
                {
                    if (Boolean.TryParse(sliceOnImportData, out s_Setting.sliceOnImport))
                    {
                        m_CustomDataProvider.GetData(SpriteEditorMenuSetting.kSliceSettingsKey, out var sliceSettingsData);
                        if (!String.IsNullOrEmpty(sliceSettingsData))
                        {
                            try
                            {
                                var menuSetting = JsonUtility.FromJson<SpriteEditorMenuSetting>(sliceSettingsData);
                                s_SettingsObject.settings = menuSetting;
                            }
                            catch (Exception)
                            {
                                Debug.LogError($"Texture ({m_TextureDataProvider.texture.name}) has invalid slice settings serialized: {sliceSettingsData}");
                            }
                        }
                    }
                }
            }

            buttonRect = GUIUtility.GUIToScreenRect(buttonRect);
            const float windowHeight = 255f;
            var windowSize = new Vector2(300, windowHeight);
            ShowAsDropDown(buttonRect, windowSize);

            Undo.undoRedoPerformed += UndoRedoPerformed;

            RectSettingsDirty();
        }

        private void UndoRedoPerformed()
        {
            Repaint();
        }

        void OnEnable()
        {
            AssemblyReloadEvents.beforeAssemblyReload += Close;
        }

        private void OnDisable()
        {
            AssemblyReloadEvents.beforeAssemblyReload -= Close;
            Undo.undoRedoPerformed -= UndoRedoPerformed;
            s_LastClosedTime = DateTime.Now.Ticks / TimeSpan.TicksPerMillisecond;
            m_SpriteFrameModule.potentialRects = null;
            m_SpriteFrameModule.spriteEditor.RequestRepaint();
            m_SpriteFrameModule.onModuleDeactivated -= OnModuleDeactivate;
        }

        void OnModuleDeactivate()
        {
            this.Close();
        }

        private void RectSettingsDirty()
        {
            m_SpriteFrameModule.potentialRects = GetPotentialRects();
            m_SpriteFrameModule.spriteEditor.RequestRepaint();
        }

        internal static bool ShowAtPosition(Rect buttonRect
            , SpriteFrameModule sf
            , ITextureDataProvider textureProvider
            , ISpriteCustomDataProvider customDataProvider
            , ISpriteFrameEditCapability capability)
        {
            // We could not use realtimeSinceStartUp since it is set to 0 when entering/exitting playmode, we assume an increasing time when comparing time.
            long nowMilliSeconds = DateTime.Now.Ticks / TimeSpan.TicksPerMillisecond;
            bool justClosed = nowMilliSeconds < s_LastClosedTime + 50;
            if (!justClosed)
            {
                if (UnityEvent.current != null) // Event.current can be null during integration test
                    UnityEvent.current.Use();

                SpriteEditorMenu spriteEditorMenu = CreateInstance<SpriteEditorMenu>();
                spriteEditorMenu.Init(buttonRect, sf, textureProvider, customDataProvider, capability);
                return true;
            }
            return false;
        }

        private void OnGUI()
        {
            if (s_Styles == null)
                s_Styles = new Styles();

            // Leave some space above the elements
            GUILayout.Space(4);

            EditorGUIUtility.labelWidth = 124f;
            EditorGUIUtility.wideMode = true;

            GUI.Label(new Rect(0, 0, position.width, position.height), GUIContent.none, s_Styles.background);

            using (new EditorGUI.DisabledScope(m_CustomDataProvider == null || !m_CanSliceOnImport))
            {
                EditorGUI.BeginChangeCheck();
                s_Setting.sliceOnImport = EditorGUILayout.Toggle(s_Styles.sliceOnImportLabel, s_Setting.sliceOnImport);
                if (EditorGUI.EndChangeCheck() && m_CustomDataProvider != null)
                {
                    m_CustomDataProvider.SetData(SpriteEditorMenuSetting.kSliceOnImportKey, s_Setting.sliceOnImport.ToString());
                    m_SpriteFrameModule.spriteEditor.SetDataModified();
                }
            }

            EditorGUI.BeginChangeCheck();
            SpriteEditorMenuSetting.SlicingType slicingType = s_Setting.slicingType;
            slicingType = (SpriteEditorMenuSetting.SlicingType)EditorGUILayout.EnumPopup(s_Styles.typeLabel, slicingType);
            if (EditorGUI.EndChangeCheck())
            {
                Undo.RegisterCompleteObjectUndo(s_SettingsObject, "Change slicing type");
                s_Setting.slicingType = slicingType;

                UpdateToDefaultAutoSliceMethod();
                RectSettingsDirty();
            }
            switch (slicingType)
            {
                case SpriteEditorMenuSetting.SlicingType.GridByCellSize:
                case SpriteEditorMenuSetting.SlicingType.GridByCellCount:
                    OnGridGUI();
                    break;
                case SpriteEditorMenuSetting.SlicingType.Automatic:
                    OnAutomaticGUI();
                    break;
                case SpriteEditorMenuSetting.SlicingType.IsometricGrid:
                    OnIsometricGridGUI();
                    break;
            }

            DoPivotGUI();
            GUILayout.Space(2f);
            EditorGUI.BeginChangeCheck();
            var slicingMethod = s_Setting.autoSlicingMethod;
            slicingMethod = EditorGUILayout.Popup(s_Styles.methodLabel, slicingMethod, s_Styles.slicingMethodOptions);
            if (EditorGUI.EndChangeCheck())
            {
                Undo.RegisterCompleteObjectUndo(s_SettingsObject, "Change Slicing Method");
                s_Setting.autoSlicingMethod = slicingMethod;
            }

            EditorGUILayout.HelpBox(s_Styles.slicingMethodInfoText[slicingMethod], MessageType.Info);

            GUILayout.FlexibleSpace();
            GUILayout.BeginHorizontal();
            GUILayout.Space(EditorGUIUtility.labelWidth + 4);
            if (GUILayout.Button(s_Styles.sliceButtonLabel))
            {
                if (DoesNotNeedWarning() || EditorUtility.DisplayDialog(s_Styles.deleteExistingTitle,
                        s_Styles.deleteExistingMessage, s_Styles.yes, s_Styles.no))
                {
                    DoSlicing();
                    if (m_CustomDataProvider != null)
                    {
                        var sliceSettingsData = JsonUtility.ToJson(s_Setting);
                        m_CustomDataProvider.SetData(SpriteEditorMenuSetting.kSliceSettingsKey, sliceSettingsData);
                    }
                }
            }
            GUILayout.EndHorizontal();
        }

        private bool DoesNotNeedWarning()
        {
            var hasNoData = m_SpriteFrameModule.spriteCount == 0;
            var isNotUsingDeleteAll = s_Setting.autoSlicingMethod != (int)SpriteFrameModule.AutoSlicingMethod.DeleteAll;
            if (hasNoData || isNotUsingDeleteAll)
                return true;

            var onlyUsingDefaultName = m_SpriteFrameModule.IsOnlyUsingDefaultNamedSpriteRects();
            return onlyUsingDefaultName;
        }

        private static void UpdateToDefaultAutoSliceMethod()
        {
            s_Setting.autoSlicingMethod = (int)SpriteFrameModule.AutoSlicingMethod.DeleteAll;
        }

        private void DoSlicing()
        {
            switch (s_Setting.slicingType)
            {
                case SpriteEditorMenuSetting.SlicingType.GridByCellCount:
                case SpriteEditorMenuSetting.SlicingType.GridByCellSize:
                    DoGridSlicing();
                    break;
                case SpriteEditorMenuSetting.SlicingType.Automatic:
                    DoAutomaticSlicing();
                    break;
                case SpriteEditorMenuSetting.SlicingType.IsometricGrid:
                    DoIsometricGridSlicing();
                    break;
            }
        }

        private void TwoIntFields(GUIContent label, GUIContent labelX, GUIContent labelY, ref int x, ref int y)
        {
            float height = EditorGUI.kSingleLineHeight;
            Rect rect = GUILayoutUtility.GetRect(EditorGUILayout.kLabelFloatMinW, EditorGUILayout.kLabelFloatMaxW, height, height, EditorStyles.numberField);

            Rect labelRect = rect;
            labelRect.width = EditorGUIUtility.labelWidth;
            labelRect.height = EditorGUI.kSingleLineHeight;

            GUI.Label(labelRect, label);

            Rect fieldRect = rect;
            fieldRect.width -= EditorGUIUtility.labelWidth;
            fieldRect.height = EditorGUI.kSingleLineHeight;
            fieldRect.x += EditorGUIUtility.labelWidth;
            fieldRect.width /= 2;
            fieldRect.width -= 2;

            EditorGUIUtility.labelWidth = 12;

            x = EditorGUI.IntField(fieldRect, labelX, x);
            fieldRect.x += fieldRect.width + 3;
            y = EditorGUI.IntField(fieldRect, labelY, y);

            EditorGUIUtility.labelWidth = labelRect.width;
        }

        private void OnGridGUI()
        {
            int width, height;
            m_TextureDataProvider.GetTextureActualWidthAndHeight(out width, out height);
            var texture = m_TextureDataProvider.GetReadableTexture2D();
            int maxWidth = texture != null ? width : 4096;
            int maxHeight = texture != null ? height : 4096;

            if (s_Setting.slicingType == SpriteEditorMenuSetting.SlicingType.GridByCellCount)
            {
                int x = (int)s_Setting.gridCellCount.x;
                int y = (int)s_Setting.gridCellCount.y;

                EditorGUI.BeginChangeCheck();
                TwoIntFields(s_Styles.columnAndRowLabel, s_Styles.columnLabel, s_Styles.rowLabel, ref x, ref y);
                if (EditorGUI.EndChangeCheck())
                {
                    Undo.RegisterCompleteObjectUndo(s_SettingsObject, "Change column & row");

                    s_Setting.gridCellCount.x = Mathf.Clamp(x, 1, maxWidth);
                    s_Setting.gridCellCount.y = Mathf.Clamp(y, 1, maxHeight);
                    RectSettingsDirty();
                }
            }
            else
            {
                int x = (int)s_Setting.gridSpriteSize.x;
                int y = (int)s_Setting.gridSpriteSize.y;

                EditorGUI.BeginChangeCheck();
                TwoIntFields(s_Styles.pixelSizeLabel, s_Styles.xLabel, s_Styles.yLabel, ref x, ref y);
                if (EditorGUI.EndChangeCheck())
                {
                    Undo.RegisterCompleteObjectUndo(s_SettingsObject, "Change grid size");

                    s_Setting.gridSpriteSize.x = Mathf.Clamp(x, 1, maxWidth);
                    s_Setting.gridSpriteSize.y = Mathf.Clamp(y, 1, maxHeight);
                    RectSettingsDirty();
                }
            }

            {
                int x = (int)s_Setting.gridSpriteOffset.x;
                int y = (int)s_Setting.gridSpriteOffset.y;

                EditorGUI.BeginChangeCheck();
                TwoIntFields(s_Styles.offsetLabel, s_Styles.xLabel, s_Styles.yLabel, ref x, ref y);
                if (EditorGUI.EndChangeCheck())
                {
                    Undo.RegisterCompleteObjectUndo(s_SettingsObject, "Change grid offset");

                    s_Setting.gridSpriteOffset.x = Mathf.Clamp(x, 0, maxWidth - s_Setting.gridSpriteSize.x);
                    s_Setting.gridSpriteOffset.y = Mathf.Clamp(y, 0, maxHeight - s_Setting.gridSpriteSize.y);
                    RectSettingsDirty();
                }
            }

            {
                int x = (int)s_Setting.gridSpritePadding.x;
                int y = (int)s_Setting.gridSpritePadding.y;

                EditorGUI.BeginChangeCheck();
                TwoIntFields(s_Styles.paddingLabel, s_Styles.xLabel, s_Styles.yLabel, ref x, ref y);
                if (EditorGUI.EndChangeCheck())
                {
                    Undo.RegisterCompleteObjectUndo(s_SettingsObject, "Change grid padding");

                    s_Setting.gridSpritePadding.x = Mathf.Clamp(x, 0, maxWidth);
                    s_Setting.gridSpritePadding.y = Mathf.Clamp(y, 0, maxHeight);
                    RectSettingsDirty();
                }
            }

            EditorGUI.BeginChangeCheck();
            bool keepEmptyRects = s_Setting.keepEmptyRects;
            keepEmptyRects = EditorGUILayout.Toggle(s_Styles.keepEmptyRectsLabel, keepEmptyRects);
            if (EditorGUI.EndChangeCheck())
            {
                Undo.RegisterCompleteObjectUndo(s_SettingsObject, "Keep Empty Rects");
                s_Setting.keepEmptyRects = keepEmptyRects;
            }
        }

        private void OnAutomaticGUI()
        {
            var texture = m_TextureDataProvider.texture;
            if (texture != null && GraphicsFormatUtility.IsCompressedFormat(texture.format))
            {
                EditorGUILayout.LabelField(s_Styles.automaticSlicingHintLabel, s_Styles.notice);
            }
        }

        private void OnIsometricGridGUI()
        {
            int width, height;
            m_TextureDataProvider.GetTextureActualWidthAndHeight(out width, out height);
            var texture = m_TextureDataProvider.GetReadableTexture2D();
            int maxWidth = texture != null ? width : 4096;
            int maxHeight = texture != null ? height : 4096;

            {
                int x = (int)s_Setting.gridSpriteSize.x;
                int y = (int)s_Setting.gridSpriteSize.y;

                EditorGUI.BeginChangeCheck();
                TwoIntFields(s_Styles.pixelSizeLabel, s_Styles.xLabel, s_Styles.yLabel, ref x, ref y);
                if (EditorGUI.EndChangeCheck())
                {
                    Undo.RegisterCompleteObjectUndo(s_SettingsObject, "Change grid size");

                    s_Setting.gridSpriteSize.x = Mathf.Clamp(x, 1, maxWidth);
                    s_Setting.gridSpriteSize.y = Mathf.Clamp(y, 1, maxHeight);
                    RectSettingsDirty();
                }
            }

            {
                int x = (int)s_Setting.gridSpriteOffset.x;
                int y = (int)s_Setting.gridSpriteOffset.y;

                EditorGUI.BeginChangeCheck();
                TwoIntFields(s_Styles.offsetLabel, s_Styles.xLabel, s_Styles.yLabel, ref x, ref y);
                if (EditorGUI.EndChangeCheck())
                {
                    Undo.RegisterCompleteObjectUndo(s_SettingsObject, "Change grid offset");

                    s_Setting.gridSpriteOffset.x = Mathf.Clamp(x, 0, maxWidth - s_Setting.gridSpriteSize.x);
                    s_Setting.gridSpriteOffset.y = Mathf.Clamp(y, 0, maxHeight - s_Setting.gridSpriteSize.y);
                    RectSettingsDirty();
                }
            }

            EditorGUI.BeginChangeCheck();
            bool keepEmptyRects = s_Setting.keepEmptyRects;
            keepEmptyRects = EditorGUILayout.Toggle(s_Styles.keepEmptyRectsLabel, keepEmptyRects);
            if (EditorGUI.EndChangeCheck())
            {
                Undo.RegisterCompleteObjectUndo(s_SettingsObject, "Keep Empty Rects");
                s_Setting.keepEmptyRects = keepEmptyRects;
            }
            EditorGUI.BeginChangeCheck();
            bool isAlternate = s_Setting.isAlternate;
            isAlternate = EditorGUILayout.Toggle(s_Styles.isAlternateLabel, isAlternate);
            if (EditorGUI.EndChangeCheck())
            {
                Undo.RegisterCompleteObjectUndo(s_SettingsObject, "Is Alternate");
                s_Setting.isAlternate = isAlternate;
                RectSettingsDirty();
            }
        }

        private void DoPivotGUI()
        {
            EditorGUI.BeginChangeCheck();
            int alignment = s_Setting.spriteAlignment;
            alignment = EditorGUILayout.Popup(s_Styles.pivotLabel, alignment, s_Styles.spriteAlignmentOptions);
            if (EditorGUI.EndChangeCheck())
            {
                Undo.RegisterCompleteObjectUndo(s_SettingsObject, "Change Alignment");
                s_Setting.spriteAlignment = alignment;
                s_Setting.pivot = SpriteEditorUtility.GetPivotValue((SpriteAlignment)alignment, s_Setting.pivot);
            }

            Vector2 pivot = s_Setting.pivot;
            EditorGUI.BeginChangeCheck();
            using (new EditorGUI.DisabledScope(alignment != (int)SpriteAlignment.Custom))
            {
                pivot = EditorGUILayout.Vector2Field(s_Styles.customPivotLabel, pivot);
            }
            if (EditorGUI.EndChangeCheck())
            {
                Undo.RegisterCompleteObjectUndo(s_SettingsObject, "Change custom pivot");

                s_Setting.pivot = pivot;
            }
        }

        private void DoAutomaticSlicing()
        {
            // 4 seems to be a pretty nice min size for a automatic sprite slicing. It used to be exposed to the slicing dialog, but it is actually better workflow to slice&crop manually than find a suitable size number
            m_SpriteFrameModule.DoAutomaticSlicing(4, s_Setting.spriteAlignment, s_Setting.pivot, (SpriteFrameModule.AutoSlicingMethod)s_Setting.autoSlicingMethod);
        }

        private void DoGridSlicing()
        {
            if (s_Setting.slicingType == SpriteEditorMenuSetting.SlicingType.GridByCellCount)
                SetGridCellSizeWithCellCount();

            m_SpriteFrameModule.DoGridSlicing(s_Setting.gridSpriteSize, s_Setting.gridSpriteOffset, s_Setting.gridSpritePadding, s_Setting.spriteAlignment, s_Setting.pivot, (SpriteFrameModule.AutoSlicingMethod)s_Setting.autoSlicingMethod, s_Setting.keepEmptyRects);
        }

        private void DoIsometricGridSlicing()
        {
            m_SpriteFrameModule.DoIsometricGridSlicing(s_Setting.gridSpriteSize, s_Setting.gridSpriteOffset, s_Setting.spriteAlignment, s_Setting.pivot, (SpriteFrameModule.AutoSlicingMethod)s_Setting.autoSlicingMethod, s_Setting.keepEmptyRects, s_Setting.isAlternate);
        }

        private void SetGridCellSizeWithCellCount()
        {
            DetermineGridCellSizeWithCellCount(out var cellSize);
            s_Setting.gridSpriteSize = cellSize;
        }

        private void DetermineGridCellSizeWithCellCount(out Vector2 cellSize)
        {
            m_TextureDataProvider.GetTextureActualWidthAndHeight(out var width, out var height);
            var texture = m_TextureDataProvider.GetReadableTexture2D();
            int maxWidth = texture != null ? width : 4096;
            int maxHeight = texture != null ? height : 4096;

            SpriteEditorUtility.DetermineGridCellSizeWithCellCount(maxWidth, maxHeight, s_Setting.gridSpriteOffset, s_Setting.gridSpritePadding, s_Setting.gridCellCount, out cellSize);
        }
    }
}
