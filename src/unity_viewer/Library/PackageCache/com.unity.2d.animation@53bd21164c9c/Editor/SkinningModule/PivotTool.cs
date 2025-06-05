using System;
using UnityEditor.U2D.Layout;
using UnityEditor.U2D.Common;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.U2D.Animation
{
#if ENABLE_UXML_SERIALIZED_DATA
    [UxmlElement]
#endif
    internal partial class PivotInspectorPanel : VisualElement
    {
        EnumField m_PivotAlignment;
        Vector2Field m_PivotPosition;

#if ENABLE_UXML_TRAITS
        public class CustomUxmlFactory : UxmlFactory<PivotInspectorPanel, UxmlTraits> { }
#endif

        internal static PivotInspectorPanel CreateFromUxml()
        {
            var visualTree = ResourceLoader.Load<VisualTreeAsset>("SkinningModule/PivotInspectorPanel.uxml");
            var ve = (PivotInspectorPanel)visualTree.CloneTree().Q("PivotInspectorPanel");
            ve.styleSheets.Add(ResourceLoader.Load<StyleSheet>("SkinningModule/PivotInspectorPanelStyle.uss"));
            if (EditorGUIUtility.isProSkin)
                ve.AddToClassList("Dark");
            ve.LocalizeTextInChildren();
            ve.BindElements();
            return ve;
        }

        private void BindElements()
        {
            m_PivotPosition = this.Q<Vector2Field>("PivotPositionField");
            m_PivotAlignment = this.Q<EnumField>("pivotField");
            m_PivotAlignment.Init(SpriteAlignment.Center);
            m_PivotAlignment.label = TextContent.pivot;
        }

        public EnumField pivotAlignment
        {
            get => m_PivotAlignment;
            set => m_PivotAlignment = value;
        }

        public Vector2Field pivotPosition
        {
            get => m_PivotPosition;
            set => m_PivotPosition = value;
        }
    }

    internal class PivotTool : SkeletonToolWrapper
    {
        static class Styles
        {
            public static GUIStyle pivotdotactive = "U2D.pivotDotActive";
            public static GUIStyle pivotdot = "U2D.pivotDot";
        }

        Vector2 m_Pivot = Vector2.zero;
        Vector2 m_CurrentMousePosition;
        Vector2 m_DragScreenOffset;
        Vector2 m_DragStartScreenPosition;
        SpriteCache m_LastSelectedSprite;
        PivotInspectorPanel m_InspectorPanel;
        int m_SlideHashCode = "PivotTool_Slider1D".GetHashCode();
        readonly Rect k_PivotNormalizedRect = Rect.MinMaxRect(0, 0, 1, 1);
        Rect m_PivotRect = Rect.zero;

        static bool CanSelectWhileInPivotTool() => false;

        public override void Initialize(LayoutOverlay layout)
        {
            base.Initialize(layout);
            m_InspectorPanel = PivotInspectorPanel.CreateFromUxml();
            layout.rightOverlay.Add(m_InspectorPanel);
            m_InspectorPanel.SetHiddenFromLayout(true);
            m_InspectorPanel.pivotAlignment.RegisterValueChangedCallback(PivotAlignmentValueChange);
            m_InspectorPanel.pivotPosition.RegisterValueChangedCallback(PivotPositionValueChange);
        }

        void PivotAlignmentValueChange(ChangeEvent<Enum> evt)
        {
            m_Pivot = GetPivotPoint((SpriteAlignment)m_InspectorPanel.pivotAlignment.value, m_Pivot);
            m_InspectorPanel.pivotPosition.SetValueWithoutNotify(m_Pivot);
            UpdateCharacterPivot();
        }

        void PivotPositionValueChange(ChangeEvent<Vector2> evt)
        {
            m_Pivot = m_InspectorPanel.pivotPosition.value;
            m_InspectorPanel.pivotAlignment.SetValueWithoutNotify(SpriteAlignment.Custom);
            UpdateCharacterPivot();
        }


        protected override void OnGUI()
        {
            base.OnGUI();
            var pivot = PivotSlider(m_PivotRect, m_Pivot, Styles.pivotdot, Styles.pivotdotactive);
            if (m_Pivot != pivot)
            {
                UpdateViewFields();
                m_Pivot = pivot;
                UpdateCharacterPivot();
            }
        }

        void UpdateCharacterPivot()
        {
            using (skinningCache.UndoScope(TextContent.pivotChanged))
            {
                skinningCache.character.pivot = m_Pivot;
                skinningCache.events.pivotChange.Invoke();
            }
        }

        protected override void OnActivate()
        {
            if (skinningCache.hasCharacter)
            {
                base.OnActivate();
                m_PivotRect = new Rect(0, 0, skinningCache.character.dimension.x, skinningCache.character.dimension.y);
                m_LastSelectedSprite = skinningCache.selectedSprite;
                m_InspectorPanel.SetHiddenFromLayout(false);
                m_Pivot = skinningCache.character.pivot;
                UpdateViewFields();

                skinningCache.selectionTool.CanSelect += CanSelectWhileInPivotTool;
                skinningCache.selectedSprite = null;
            }
            else
            {
                m_InspectorPanel.SetHiddenFromLayout(true);
            }
        }

        void UpdateViewFields()
        {
            SpriteAlignment alignment;
            TranslatePivotPoint(m_Pivot, out alignment);
            m_InspectorPanel.pivotAlignment.SetValueWithoutNotify(alignment);
            m_InspectorPanel.pivotPosition.SetValueWithoutNotify(m_Pivot);
        }

        protected override void OnDeactivate()
        {
            if (skinningCache.hasCharacter)
            {
                base.OnDeactivate();
                m_InspectorPanel.SetHiddenFromLayout(true);

                skinningCache.selectionTool.CanSelect -= CanSelectWhileInPivotTool;
                if (isActive)
                    skinningCache.selectedSprite = m_LastSelectedSprite;
            }
        }

        void TranslatePivotPoint(Vector2 pivot, out SpriteAlignment alignment)
        {
            if (new Vector2(k_PivotNormalizedRect.xMin, k_PivotNormalizedRect.yMax) == pivot)
                alignment = SpriteAlignment.TopLeft;
            else if (new Vector2(k_PivotNormalizedRect.center.x, k_PivotNormalizedRect.yMax) == pivot)
                alignment = SpriteAlignment.TopCenter;
            else if (new Vector2(k_PivotNormalizedRect.xMax, k_PivotNormalizedRect.yMax) == pivot)
                alignment = SpriteAlignment.TopRight;
            else if (new Vector2(k_PivotNormalizedRect.xMin, k_PivotNormalizedRect.center.y) == pivot)
                alignment = SpriteAlignment.LeftCenter;
            else if (new Vector2(k_PivotNormalizedRect.center.x, k_PivotNormalizedRect.center.y) == pivot)
                alignment = SpriteAlignment.Center;
            else if (new Vector2(k_PivotNormalizedRect.xMax, k_PivotNormalizedRect.center.y) == pivot)
                alignment = SpriteAlignment.RightCenter;
            else if (new Vector2(k_PivotNormalizedRect.xMin, k_PivotNormalizedRect.yMin) == pivot)
                alignment = SpriteAlignment.BottomLeft;
            else if (new Vector2(k_PivotNormalizedRect.center.x, k_PivotNormalizedRect.yMin) == pivot)
                alignment = SpriteAlignment.BottomCenter;
            else if (new Vector2(k_PivotNormalizedRect.xMax, k_PivotNormalizedRect.yMin) == pivot)
                alignment = SpriteAlignment.BottomRight;
            else
                alignment = SpriteAlignment.Custom;
        }

        Vector2 GetPivotPoint(SpriteAlignment alignment, Vector2 customPivot)
        {
            switch (alignment)
            {
                case SpriteAlignment.TopLeft:
                    return new Vector2(k_PivotNormalizedRect.xMin, k_PivotNormalizedRect.yMax);

                case SpriteAlignment.TopCenter:
                    return new Vector2(k_PivotNormalizedRect.center.x, k_PivotNormalizedRect.yMax);

                case SpriteAlignment.TopRight:
                    return new Vector2(k_PivotNormalizedRect.xMax, k_PivotNormalizedRect.yMax);

                case SpriteAlignment.LeftCenter:
                    return new Vector2(k_PivotNormalizedRect.xMin, k_PivotNormalizedRect.center.y);

                case SpriteAlignment.Center:
                    return new Vector2(k_PivotNormalizedRect.center.x, k_PivotNormalizedRect.center.y);

                case SpriteAlignment.RightCenter:
                    return new Vector2(k_PivotNormalizedRect.xMax, k_PivotNormalizedRect.center.y);

                case SpriteAlignment.BottomLeft:
                    return new Vector2(k_PivotNormalizedRect.xMin, k_PivotNormalizedRect.yMin);

                case SpriteAlignment.BottomCenter:
                    return new Vector2(k_PivotNormalizedRect.center.x, k_PivotNormalizedRect.yMin);

                case SpriteAlignment.BottomRight:
                    return new Vector2(k_PivotNormalizedRect.xMax, k_PivotNormalizedRect.yMin);

                case SpriteAlignment.Custom:
                    return new Vector2(customPivot.x * k_PivotNormalizedRect.width, customPivot.y * k_PivotNormalizedRect.height);
            }

            return Vector2.zero;
        }

        Vector2 PivotSlider(Rect sprite, Vector2 pos, GUIStyle pivotDot, GUIStyle pivotDotActive)
        {
            int id = GUIUtility.GetControlID(m_SlideHashCode, FocusType.Keyboard);

            // Convert from normalized space to texture space
            pos = new Vector2(sprite.xMin + sprite.width * pos.x, sprite.yMin + sprite.height * pos.y);

            Vector2 screenVal = Handles.matrix.MultiplyPoint(pos);

            Rect handleScreenPos = new Rect(
                screenVal.x - pivotDot.fixedWidth * .5f,
                screenVal.y - pivotDot.fixedHeight * .5f,
                pivotDotActive.fixedWidth,
                pivotDotActive.fixedHeight
            );

            var evt = Event.current;
            switch (evt.GetTypeForControl(id))
            {
                case EventType.MouseDown:
                    // am I closest to the thingy?
                    if (evt.button == 0 && handleScreenPos.Contains(Event.current.mousePosition) && !evt.alt)
                    {
                        GUIUtility.hotControl = GUIUtility.keyboardControl = id; // Grab mouse focus
                        m_CurrentMousePosition = evt.mousePosition;
                        m_DragStartScreenPosition = evt.mousePosition;
                        Vector2 rectScreenCenter = Handles.matrix.MultiplyPoint(pos);
                        m_DragScreenOffset = m_CurrentMousePosition - rectScreenCenter;
                        evt.Use();
                        EditorGUIUtility.SetWantsMouseJumping(1);
                    }

                    break;
                case EventType.MouseDrag:
                    if (GUIUtility.hotControl == id)
                    {
                        m_CurrentMousePosition += evt.delta;
                        Vector2 oldPos = pos;
                        Vector3 scrPos = Handles.inverseMatrix.MultiplyPoint(m_CurrentMousePosition - m_DragScreenOffset);
                        pos = new Vector2(scrPos.x, scrPos.y);
                        if (!Mathf.Approximately((oldPos - pos).magnitude, 0f))
                            GUI.changed = true;
                        evt.Use();
                    }

                    break;
                case EventType.MouseUp:
                    if (GUIUtility.hotControl == id && (evt.button == 0 || evt.button == 2))
                    {
                        GUIUtility.hotControl = 0;
                        evt.Use();
                        EditorGUIUtility.SetWantsMouseJumping(0);
                    }

                    break;
                case EventType.KeyDown:
                    if (GUIUtility.hotControl == id)
                    {
                        if (evt.keyCode == KeyCode.Escape)
                        {
                            pos = Handles.inverseMatrix.MultiplyPoint(m_DragStartScreenPosition - m_DragScreenOffset);
                            GUIUtility.hotControl = 0;
                            GUI.changed = true;
                            evt.Use();
                        }
                    }

                    break;
                case EventType.Repaint:
                    EditorGUIUtility.AddCursorRect(handleScreenPos, MouseCursor.Arrow, id);

                    if (GUIUtility.hotControl == id)
                        pivotDotActive.Draw(handleScreenPos, GUIContent.none, id);
                    else
                        pivotDot.Draw(handleScreenPos, GUIContent.none, id);

                    break;
            }

            // Convert from texture space back to normalized space
            pos = new Vector2((pos.x - sprite.xMin) / sprite.width, (pos.y - sprite.yMin) / sprite.height);

            return pos;
        }
    }
}
