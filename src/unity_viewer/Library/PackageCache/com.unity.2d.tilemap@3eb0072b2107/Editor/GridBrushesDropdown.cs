using System;
using UnityEngine;

namespace UnityEditor.Tilemaps
{
    internal class GridBrushesDropdown : PopupWindowContent
    {
        class Styles
        {
            public GUIStyle menuItem = "MenuItem";
        }
        static Styles s_Styles;

        MenuItemProvider m_ItemProvider;
        FlexibleMenuModifyItemUI m_ModifyItemUI;
        readonly Action<int, object> m_ItemClickedCallback;
        Vector2 m_ScrollPosition = Vector2.zero;
        bool m_ShowAddNewPresetItem;
        int m_HoverIndex;
        int[] m_SeperatorIndices;
        float m_CachedWidth = -1f;
        float m_MinTextWidth = 200f;

        const float LineHeight = 18f;
        const float SeparatorHeight = 8f;
        int maxIndex { get { return m_ShowAddNewPresetItem ? m_ItemProvider.Count() : m_ItemProvider.Count() - 1; } }
        public int selectedIndex { get; set; }
        protected float minTextWidth { get { return m_MinTextWidth; } set { m_MinTextWidth = value; ClearCachedWidth(); } }

        internal class MenuItemProvider : IFlexibleMenuItemProvider
        {
            public int Count()
            {
                return GridPaletteBrushes.brushes.Count;
            }

            public object GetItem(int index)
            {
                return GridPaletteBrushes.brushes[index];
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
                return GridPaletteBrushes.brushNames[index];
            }

            public string GetTooltip(int index)
            {
                return GridPaletteBrushes.brushTooltips[index];
            }

            public bool IsModificationAllowed(int index)
            {
                return false;
            }

            public int[] GetSeperatorIndices()
            {
                return new int[0];
            }
        }

        public GridBrushesDropdown(Action<int, object> itemClickedCallback, float minWidth)
        {
            m_ItemProvider = new MenuItemProvider();
            m_ModifyItemUI = null;
            m_ItemClickedCallback = itemClickedCallback;
            m_SeperatorIndices = m_ItemProvider.GetSeperatorIndices();
            selectedIndex = GridPaletteBrushes.brushes.IndexOf(GridPaintingState.gridBrush);
            m_ShowAddNewPresetItem = m_ModifyItemUI != null;
            m_MinTextWidth = minWidth;
        }

        public override Vector2 GetWindowSize()
        {
            return CalcSize();
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
                    Rect fullRect = new Rect(0, curY, rect.width, LineHeight);
                    Rect itemRect = fullRect;
                    bool addSeparator = Array.IndexOf(m_SeperatorIndices, i) >= 0;

                    // Handle event
                    switch (evt.type)
                    {
                        case EventType.Repaint:
                            bool hover = false;
                            if (m_HoverIndex == i)
                            {
                                if (fullRect.Contains(evt.mousePosition))
                                    hover = true;
                                else
                                    m_HoverIndex = -1;
                            }

                            var tooltip = m_ItemProvider.GetTooltip(i);
                            if (!String.IsNullOrWhiteSpace(tooltip))
                                EditorGUI.LabelField(itemRect, GUIContent.Temp("", tooltip)); // Use empty label to overlay tooltip
                            s_Styles.menuItem.Draw(itemRect, GUIContent.Temp(m_ItemProvider.GetName(i)), hover, false, i == selectedIndex, false);

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
                            if (fullRect.Contains(evt.mousePosition))
                            {
                                if (m_HoverIndex != i)
                                {
                                    m_HoverIndex = i;
                                    Repaint();
                                }
                            }
                            else if (m_HoverIndex == i)
                            {
                                m_HoverIndex = -1;
                                Repaint();
                            }
                            break;
                    }

                    curY += LineHeight;
                    if (addSeparator)
                        curY += SeparatorHeight;
                } // end foreach item
            } GUI.EndScrollView();
        }

        void SelectItem(int index)
        {
            selectedIndex = index;
            if (m_ItemClickedCallback != null && index >= 0)
                m_ItemClickedCallback(index, m_ItemProvider.GetItem(index));
        }

        protected Vector2 CalcSize()
        {
            float height = (maxIndex + 1) * LineHeight + m_SeperatorIndices.Length * SeparatorHeight;
            if (m_CachedWidth < 0)
                m_CachedWidth = Math.Max(m_MinTextWidth, CalcWidth());
            return new Vector2(m_CachedWidth, height);
        }

        void ClearCachedWidth()
        {
            m_CachedWidth = -1f;
        }

        float CalcWidth()
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

        void Repaint()
        {
            HandleUtility.Repaint(); // repaints current guiview (needs rename)
        }
    }
}
