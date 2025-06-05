using System.IO;
using UnityEngine;

namespace UnityEditor.U2D.Sprites
{
    internal partial class SpriteFrameModule : SpriteFrameModuleBase
    {
        class SpriteFrameModuleStyles
        {
            public GUIContent capabilityButtonLabel = null;
            public readonly GUIContent sliceButtonLabel = EditorGUIUtility.TrTextContent("Slice", "Sprite creation and deletion is enabled.");
            public readonly GUIContent sliceButtonLabelDisabled = EditorGUIUtility.TrTextContent("Slice","Sprite creation and deletion is disabled.");
            public readonly GUIContent trimButtonLabel = EditorGUIUtility.TrTextContent("Trim", "Trims selected rectangle (T)");
            public readonly GUIContent trimButtonLabelDisabled = EditorGUIUtility.TrTextContent("Trim", "Trims selected rectangle (T). Disabled because the Sprite Position field is locked.");
            static SpriteFrameModuleStyles s_Style;

            public static SpriteFrameModuleStyles instance
            {
                get
                {
                    if (s_Style == null)
                        s_Style = new SpriteFrameModuleStyles();
                    return s_Style;
                }
            }
            public SpriteFrameModuleStyles()
            {
                var iconPath = "Packages/com.unity.2d.sprite/Editor/Assets/LockedOptions";
                if (EditorGUIUtility.isProSkin)
                {
                    var newName = "d_" + Path.GetFileName(iconPath);
                    var iconDirName = Path.GetDirectoryName(iconPath);
                    if (!string.IsNullOrEmpty(iconDirName))
                        newName = $"{iconDirName}/{newName}";

                    iconPath = newName;
                }

                if (EditorGUIUtility.pixelsPerPoint > 1)
                    iconPath = $"{iconPath}@2x";

                capabilityButtonLabel = new GUIContent(EditorGUIUtility.Load(iconPath + ".png") as Texture2D, "Toggle Sprite Editor locks.");
            }
        }

        // overrides for SpriteFrameModuleBase
        public override void DoMainGUI()
        {
            // Beautification. Run ValidateSpriteRects only when window is presented.
            if(!m_SpriteRectValidated)
            {
                EditorApplication.delayCall += ValidateSpriteRects;
            }

            // Do nothing when extension is activated.
            if (m_CurrentMode != null)
            {
                m_CurrentMode.DoMainGUI();
                return;
            }
            base.DoMainGUI();
            DrawSpriteRectGizmos();
            DrawPotentialSpriteRectGizmos();

            if (!spriteEditor.editingDisabled)
            {
                HandleGizmoMode();

                if (containsMultipleSprites)
                    HandleRectCornerScalingHandles();

                HandleBorderCornerScalingHandles();
                HandleBorderSidePointScalingSliders();

                if (containsMultipleSprites)
                    HandleRectSideScalingHandles();

                HandleBorderSideScalingHandles();
                HandlePivotHandle();

                if (containsMultipleSprites)
                    HandleDragging();

                spriteEditor.HandleSpriteSelection();

                if (containsMultipleSprites && m_CurrentEditEditCapability.data.HasCapability(EEditCapability.CreateAndDeleteSprite))
                {
                    HandleCreate();
                    HandleDelete();
                    HandleDuplicate();
                }
                spriteEditor.spriteRects = m_RectsCache.GetSpriteRects();
            }
        }

        private void DrawPotentialSpriteRectGizmos()
        {
            if (m_PotentialRects != null && m_PotentialRects.Count > 0)
                DrawRectGizmos(m_PotentialRects, Color.red);
        }

        public override void DoToolbarGUI(Rect toolbarRect)
        {
            using (new EditorGUI.DisabledScope(!containsMultipleSprites || spriteEditor.editingDisabled || m_TextureDataProvider.GetReadableTexture2D() == null || m_CurrentMode != null))
            {
                GUIStyle skin = EditorStyles.toolbarPopup;

                Rect drawArea = toolbarRect;

                var canCreateSprite = m_CurrentEditEditCapability.data.HasCapability(EEditCapability.CreateAndDeleteSprite);
                drawArea.width = skin.CalcSize(new GUIContent(SpriteFrameModuleStyles.instance.sliceButtonLabel)).x;
                SpriteUtilityWindow.DrawToolBarWidget(ref drawArea, ref toolbarRect, (adjustedDrawArea) =>
                {
                    using (new EditorGUI.DisabledScope(!canCreateSprite))
                    {
                       if (GUI.Button(adjustedDrawArea,
                               canCreateSprite ? SpriteFrameModuleStyles.instance.sliceButtonLabel : SpriteFrameModuleStyles.instance.sliceButtonLabelDisabled,
                               skin))
                       {
                           if (SpriteEditorMenu.ShowAtPosition(adjustedDrawArea, this, this, m_CustomDataProvider, m_FrameEditCapability))
                               GUIUtility.ExitGUI();
                       }
                    }
                });

                // Trim Button
                var canEditPosition = m_CurrentEditEditCapability.data.HasCapability(EEditCapability.EditSpriteRect);
                using (new EditorGUI.DisabledScope(!hasSelected || !canEditPosition))
                {
                    drawArea.x += drawArea.width;
                    drawArea.width = skin.CalcSize(SpriteFrameModuleStyles.instance.trimButtonLabel).x;
                    SpriteUtilityWindow.DrawToolBarWidget(ref drawArea, ref toolbarRect, (adjustedDrawArea) =>
                    {
                        if (GUI.Button(adjustedDrawArea,
                                canEditPosition ? SpriteFrameModuleStyles.instance.trimButtonLabel : SpriteFrameModuleStyles.instance.trimButtonLabelDisabled,
                                EditorStyles.toolbarButton))
                        {
                            TrimAlpha();
                            Repaint();
                        }
                    });
                }

                // Edit Capability button
                drawArea.x += drawArea.width;
                var expectedSize = skin.CalcSize(new GUIContent("")).x;
                drawArea.width = expectedSize + drawArea.height;
                drawArea.x = drawArea.x + toolbarRect.width - drawArea.width;
                SpriteUtilityWindow.DrawToolBarWidget(ref drawArea, ref toolbarRect, (adjustedDrawArea) =>
                {
                    if(adjustedDrawArea.width < expectedSize)
                        return;
                    if (GUI.Button(adjustedDrawArea, SpriteFrameModuleStyles.instance.capabilityButtonLabel, skin))
                    {
                        if (SpriteFrameCapabilityWindow.ShowAtPosition(adjustedDrawArea, m_CurrentEditEditCapability, Path.GetExtension(spriteAssetPath), OnEditCapabilityChanged, (a, b) =>
                            {
                                if(b)
                                    m_OnUndoCallback += a;
                                else
                                    m_OnUndoCallback -= a;
                            }))
                            GUIUtility.ExitGUI();
                    }
                });
            }
        }

        private void HandleRectCornerScalingHandles()
        {
            if (!hasSelected)
                return;

            GUIStyle dragDot = styles.dragdot;
            GUIStyle dragDotActive = styles.dragdotactive;
            var color = Color.white;

            Rect rect = new Rect(selectedSpriteRect_Rect);

            float left = rect.xMin;
            float right = rect.xMax;
            float top = rect.yMax;
            float bottom = rect.yMin;

            EditorGUI.BeginChangeCheck();

            bool canEdit = m_CurrentEditEditCapability.data.HasCapability(EEditCapability.EditSpriteRect);
            HandleBorderPointSlider(ref left, ref top,  canEdit ?  MouseCursor.ResizeUpLeft : MouseCursor.NotAllowed, false, dragDot, dragDotActive, color);
            HandleBorderPointSlider(ref right, ref top, canEdit ? MouseCursor.ResizeUpRight : MouseCursor.NotAllowed, false, dragDot, dragDotActive, color);
            HandleBorderPointSlider(ref left, ref bottom, canEdit ? MouseCursor.ResizeUpRight : MouseCursor.NotAllowed, false, dragDot, dragDotActive, color);
            HandleBorderPointSlider(ref right, ref bottom, canEdit ? MouseCursor.ResizeUpLeft : MouseCursor.NotAllowed, false, dragDot, dragDotActive, color);

            if (EditorGUI.EndChangeCheck() && canEdit)
            {
                rect.xMin = left;
                rect.xMax = right;
                rect.yMax = top;
                rect.yMin = bottom;
                ScaleSpriteRect(rect);
                PopulateSpriteFrameInspectorField();
            }
        }

        private void HandleRectSideScalingHandles()
        {
            if (!hasSelected)
                return;

            Rect rect = new Rect(selectedSpriteRect_Rect);

            float left = rect.xMin;
            float right = rect.xMax;
            float top = rect.yMax;
            float bottom = rect.yMin;

            Vector2 screenRectTopLeft = Handles.matrix.MultiplyPoint(new Vector3(rect.xMin, rect.yMin));
            Vector2 screenRectBottomRight = Handles.matrix.MultiplyPoint(new Vector3(rect.xMax, rect.yMax));

            float screenRectWidth = Mathf.Abs(screenRectBottomRight.x - screenRectTopLeft.x);
            float screenRectHeight = Mathf.Abs(screenRectBottomRight.y - screenRectTopLeft.y);

            EditorGUI.BeginChangeCheck();

            bool canEdit = m_CurrentEditEditCapability.data.HasCapability(EEditCapability.EditSpriteRect);
            left = HandleBorderScaleSlider(left, rect.yMax, screenRectWidth, screenRectHeight, true, canEdit);
            right = HandleBorderScaleSlider(right, rect.yMax, screenRectWidth, screenRectHeight, true, canEdit);

            top = HandleBorderScaleSlider(rect.xMin, top, screenRectWidth, screenRectHeight, false, canEdit);
            bottom = HandleBorderScaleSlider(rect.xMin, bottom, screenRectWidth, screenRectHeight, false, canEdit);

            if (EditorGUI.EndChangeCheck() && canEdit)
            {
                rect.xMin = left;
                rect.xMax = right;
                rect.yMax = top;
                rect.yMin = bottom;

                ScaleSpriteRect(rect);
                PopulateSpriteFrameInspectorField();
            }
        }

        private void HandleDragging()
        {
            if (hasSelected && !MouseOnTopOfInspector() && m_CurrentEditEditCapability.data.HasCapability(EEditCapability.EditSpriteRect))
            {
                Rect textureBounds = new Rect(0, 0, textureActualWidth, textureActualHeight);
                EditorGUI.BeginChangeCheck();

                Rect oldRect = selectedSpriteRect_Rect;
                Rect newRect = SpriteEditorUtility.ClampedRect(SpriteEditorUtility.RoundedRect(SpriteEditorHandles.SliderRect(oldRect)), textureBounds, true);

                if (EditorGUI.EndChangeCheck())
                {
                    selectedSpriteRect_Rect = newRect;
                    UpdatePositionField(null);
                }
            }
        }

        private void HandleCreate()
        {
            if (!MouseOnTopOfInspector() && !eventSystem.current.alt)
            {
                // Create new rects via dragging in empty space
                EditorGUI.BeginChangeCheck();
                Rect newRect = SpriteEditorHandles.RectCreator(textureActualWidth, textureActualHeight, styles.createRect);
                if (EditorGUI.EndChangeCheck() && newRect.width > 0f && newRect.height > 0f)
                {
                    CreateSprite(newRect);
                    GUIUtility.keyboardControl = 0;
                }
            }
        }

        private void HandleDuplicate()
        {
            IEvent evt = eventSystem.current;
            if ((evt.type == EventType.ValidateCommand || evt.type == EventType.ExecuteCommand)
                && evt.commandName == EventCommandNames.Duplicate)
            {
                if (evt.type == EventType.ExecuteCommand)
                    DuplicateSprite();

                evt.Use();
            }
        }

        private void HandleDelete()
        {
            IEvent evt = eventSystem.current;

            if ((evt.type == EventType.ValidateCommand || evt.type == EventType.ExecuteCommand)
                && (evt.commandName == EventCommandNames.SoftDelete || evt.commandName == EventCommandNames.Delete))
            {
                if (evt.type == EventType.ExecuteCommand && hasSelected)
                    DeleteSprite();

                evt.Use();
            }
        }

        public override void DoPostGUI()
        {
            if (m_CurrentMode != null)
                m_CurrentMode.DoPostGUI();
            else
            {
                base.DoPostGUI();
            }
        }
    }
}
