using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    internal class TilePaletteActivePaletteDropdownMenu : IGenericMenu
    {
        private const float k_DropdownWidth = 156f;

        private GridPalettesDropdown m_Dropdown;
        private TilePaletteWhiteboxPaletteDropdownMenu m_Menu;

        public TilePaletteActivePaletteDropdownMenu()
        {
            int index = GridPaintingState.palettes != null ? GridPaintingState.palettes.IndexOf(GridPaintingState.palette) : 0;
            var menuData = new GridPalettesDropdown.MenuItemProvider();
            m_Dropdown = new GridPalettesDropdown(menuData, index, null, SelectPalette, HoverPalette, k_DropdownWidth);
        }

        private void HoverPalette(int index, Rect itemRect)
        {
            if (index <= GridPalettes.palettes.Count )
            {
                if (m_Menu != null)
                {
                    m_Menu.Close();
                    m_Menu = null;
                }
                return;
            }

            if (!GridPaletteWhiteboxPalettesDropdown.IsOpen)
            {
                m_Menu = new TilePaletteWhiteboxPaletteDropdownMenu(OnClose);

                var popupRect = itemRect;
                popupRect.x += itemRect.width;
                popupRect.y -= itemRect.height;

                m_Menu.DropDown(popupRect);
            }
        }

        private void OnClose()
        {
            m_Dropdown.editorWindow.Close();
        }

        public void AddItem(string itemName, bool isChecked, System.Action action)
        {
        }

        public void AddItem(string itemName, bool isChecked, System.Action<object> action, object data)
        {
        }

        public void AddDisabledItem(string itemName, bool isChecked)
        {
        }

        public void AddSeparator(string path)
        {
        }

        public void DropDown(Rect position, VisualElement targetElement = null, bool anchored = false)
        {
            PopupWindow.Show(position, m_Dropdown);
        }

        private void SelectPalette(int i, object o)
        {
            if (i < GridPaintingState.palettes.Count)
            {
                GridPaintingState.palette = GridPaintingState.palettes[i];
            }
            else if (i == GridPaintingState.palettes.Count)
            {
                m_Dropdown.editorWindow.Close();
                OpenAddPalettePopup(new Rect());
            }
        }

        private void OpenAddPalettePopup(Rect rect)
        {
            bool popupOpened = GridPaletteAddPopup.ShowAtPosition(rect);
            if (popupOpened)
                GUIUtility.ExitGUI();
        }
    }
}
