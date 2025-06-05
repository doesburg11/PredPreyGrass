using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    internal class TilePaletteBrushesDropdownMenu : IGenericMenu
    {
        private const float k_BrushDropdownWidth = 150f;

        private GridBrushesDropdown m_Dropdown;

        public TilePaletteBrushesDropdownMenu()
        {
            m_Dropdown = new GridBrushesDropdown(SelectBrush, k_BrushDropdownWidth);
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

        private void SelectBrush(int i, object o)
        {
            GridPaintingState.gridBrush = GridPaletteBrushes.brushes[i];
        }
    }
}
