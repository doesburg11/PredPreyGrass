using System;
using UnityEditor.Experimental;
using UnityEngine;

namespace UnityEditor.Tilemaps
{
    internal class GridPalettesDropdown : PopupWindowContent
    {
        private class Styles
        {
            public GUIStyle menuItem = "MenuItem";
            public GUIContent backIcon = EditorGUIUtility.TrIconContent("tab_next");
            public static readonly GUIStyle sceneVisibilityStyle = "SceneVisibility";
        }

        private static Styles s_Styles;

        internal class MenuItemProvider : IFlexibleMenuItemProvider
        {
            public int Count()
            {
                var count = GridPaintingState.palettes.Count + 1;
                if (TilePaletteWhiteboxSamplesUtility.whiteboxSamples.Count > 0)
                    count += 1;
                return count;
            }

            public object GetItem(int index)
            {
                if (index < GridPaintingState.palettes.Count)
                    return GridPaintingState.palettes[index];

                return null;
            }

            public int Add(object obj)
            {
                throw new NotImplementedException();
            }

            public void Replace(int index, object newPresetObject)
            {
                throw new NotImplementedException();
            }

            public void Remove(int index)
            {
                throw new NotImplementedException();
            }

            public object Create()
            {
                throw new NotImplementedException();
            }

            public void Move(int index, int destIndex, bool insertAfterDestIndex)
            {
                throw new NotImplementedException();
            }

            public string GetName(int index)
            {
                if (index < GridPaintingState.palettes.Count)
                    return GridPaintingState.palettes[index].name;
                else if (index == GridPaintingState.palettes.Count)
                    return "Create New Tile Palette";
                else if (index == GridPaintingState.palettes.Count + 1)
                    return "Create New Whitebox Tile Palette";
                else
                    return "";
            }

            public bool IsModificationAllowed(int index)
            {
                return false;
            }

            public int[] GetSeperatorIndices()
            {
                return new int[] { GridPaintingState.palettes.Count - 1 };
            }
        }

        private IFlexibleMenuItemProvider m_ItemProvider;
        private FlexibleMenuModifyItemUI m_ModifyItemUI;
        private readonly Action<int, object> m_ItemClickedCallback;
        private readonly Action<int, Rect> m_ItemHoveredCallback;
        private Vector2 m_ScrollPosition = Vector2.zero;
        private bool m_ShowAddNewPresetItem;
        private int m_HoverIndex;
        private int[] m_SeperatorIndices;
        private float m_CachedWidth = -1f;
        private float m_MinTextWidth;

        private const float LineHeight = 18f;
        private const float SeperatorHeight = 8f;
        private int maxIndex { get { return m_ShowAddNewPresetItem ? m_ItemProvider.Count() : m_ItemProvider.Count() - 1; } }
        public int selectedIndex { get; set; }
        protected float minTextWidth { get { return m_MinTextWidth; } set { m_MinTextWidth = value; ClearCachedWidth(); } }

        public GridPalettesDropdown(IFlexibleMenuItemProvider itemProvider
            , int selectionIndex
            , FlexibleMenuModifyItemUI modifyItemUi
            , Action<int, object> itemClickedCallback
            , Action<int, Rect> itemHoveredCallback
            , float minWidth)
        {
            m_ItemProvider = itemProvider;
            m_ModifyItemUI = modifyItemUi;
            m_ItemClickedCallback = itemClickedCallback;
            m_ItemHoveredCallback = itemHoveredCallback;
            m_SeperatorIndices = m_ItemProvider.GetSeperatorIndices();
            selectedIndex = selectionIndex;
            m_ShowAddNewPresetItem = m_ModifyItemUI != null;
            m_MinTextWidth = minWidth;
        }

        public override Vector2 GetWindowSize()
        {
            return CalcSize();
        }

        protected Vector2 CalcSize()
        {
            float height = (maxIndex + 1) * LineHeight + m_SeperatorIndices.Length * SeperatorHeight;
            if (m_CachedWidth < 0)
                m_CachedWidth = Math.Max(m_MinTextWidth, CalcWidth());
            return new Vector2(m_CachedWidth, height);
        }

        private void ClearCachedWidth()
        {
            m_CachedWidth = -1f;
        }

        private float CalcWidth()
        {
            if (s_Styles == null)
                s_Styles = new Styles();

            float maxWidth = 0;
            for (int i = 0; i < m_ItemProvider.Count(); ++i)
            {
                float w = s_Styles.menuItem.CalcSize(GUIContent.Temp(m_ItemProvider.GetName(i))).x;
                maxWidth = Mathf.Max(w, maxWidth);
            }

            const float rightMargin = 6f;
            return maxWidth + rightMargin;
        }

        private void DrawRect(Rect rect, Color color)
        {
            if (Event.current.type != EventType.Repaint)
                return;

            Color orgColor = GUI.color;
            GUI.color = GUI.color * color;
            GUI.DrawTexture(rect, EditorGUIUtility.whiteTexture);
            GUI.color = orgColor;
        }

        private void Repaint()
        {
            HandleUtility.Repaint(); // repaints current guiview (needs rename)
        }

        public override void OnGUI(Rect rect)
        {
            if (s_Styles == null)
                s_Styles = new Styles();

            Event evt = Event.current;

            Rect contentRect = new Rect(0, 0, 1, CalcSize().y);
            m_ScrollPosition = GUI.BeginScrollView(rect, m_ScrollPosition, contentRect);
            {
                float curY = 0f;
                for (int i = 0; i <= maxIndex; ++i)
                {
                    int itemControlID = i + 1000000;
                    Rect itemRect = new Rect(0, curY, rect.width, LineHeight);
                    Rect backRect = new Rect(rect.width - 16 - 1, curY, 16, LineHeight);
                    bool addSeparator = Array.IndexOf(m_SeperatorIndices, i) >= 0;

                    // Handle event
                    switch (evt.type)
                    {
                        case EventType.Repaint:
                            bool hover = false;
                            if (m_HoverIndex == i)
                            {
                                if (itemRect.Contains(evt.mousePosition))
                                {
                                    hover = true;
                                }
                                else
                                    m_HoverIndex = -1;
                            }
                            s_Styles.menuItem.Draw(itemRect, GUIContent.Temp(m_ItemProvider.GetName(i)), hover, false, i == selectedIndex, false);
                            if (TilePaletteWhiteboxSamplesUtility.whiteboxSamples.Count > 0 && i == maxIndex)
                            {
                                GUI.Button(backRect, s_Styles.backIcon, Styles.sceneVisibilityStyle);
                            }
                            if (addSeparator)
                            {
                                const float margin = 4f;
                                Rect seperatorRect = new Rect(itemRect.x + margin, itemRect.y + itemRect.height + SeperatorHeight * 0.5f, itemRect.width - 2 * margin, 1);
                                DrawRect(seperatorRect, (EditorGUIUtility.isProSkin) ? new Color(0.32f, 0.32f, 0.32f, 1.333f) : new Color(0.6f, 0.6f, 0.6f, 1.333f)); // dark : light
                            }
                            break;

                        case EventType.MouseDown:
                            if (evt.button == 0 && itemRect.Contains(evt.mousePosition))
                            {
                                GUIUtility.hotControl = itemControlID;
                                if (evt.clickCount == 1)
                                {
                                    GUIUtility.hotControl = 0;
                                    SelectItem(i);
                                    editorWindow.Close();
                                    evt.Use();
                                }
                            }
                            break;
                        case EventType.MouseUp:
                            if (GUIUtility.hotControl == itemControlID)
                            {
                                GUIUtility.hotControl = 0;
                            }
                            break;
                        case EventType.MouseMove:
                            if (itemRect.Contains(evt.mousePosition))
                            {
                                m_HoverIndex = i;
                                HoverItem(itemRect, m_HoverIndex);
                            }
                            else if (m_HoverIndex == i)
                            {
                                m_HoverIndex = -1;
                            }
                            Repaint();
                            break;
                    }

                    curY += LineHeight;
                    if (addSeparator)
                        curY += SeperatorHeight;
                } // end foreach item
            } GUI.EndScrollView();
        }

        private void SelectItem(int index)
        {
            selectedIndex = index;
            if (m_ItemClickedCallback != null && index >= 0)
                m_ItemClickedCallback(index, m_ItemProvider.GetItem(index));
        }

        private void HoverItem(Rect rect, int index)
        {
            if (m_ItemHoveredCallback != null && index >= 0)
                m_ItemHoveredCallback(index, rect);
        }
    }
}
