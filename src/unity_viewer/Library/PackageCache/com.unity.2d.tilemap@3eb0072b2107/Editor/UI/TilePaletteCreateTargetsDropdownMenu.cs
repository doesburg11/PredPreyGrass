using System;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    internal class TilePaletteCreateTargetsDropdownMenu : IGenericMenu
    {
        private const float k_DropdownWidth = 156f;

        private GridPaintCreateTargetsDropdown m_Dropdown;

        private Action m_OnClose;

        private bool m_HasTilePalette;

        public TilePaletteCreateTargetsDropdownMenu(Action onClose)
        {
            m_HasTilePalette = GridPaintingState.palette != null;
            var index = -1;
            var menuData = new GridPaintCreateTargetsDropdown.MenuItemProvider(m_HasTilePalette);
            m_Dropdown = new GridPaintCreateTargetsDropdown(menuData, index, null, CreateTilemapTarget, k_DropdownWidth);
            m_OnClose = onClose;
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
            PopupWindow.Show(position, m_Dropdown, null, ShowMode.PopupMenu);
        }

        public void Close()
        {
            PopupWindow.Show(default, m_Dropdown, null, ShowMode.PopupMenu);
        }

        private void CreateTilemapTarget(int i, object o)
        {
            m_Dropdown.editorWindow.Close();
            if (!m_HasTilePalette)
                i += 1;
            GameObjectCreation.CreateTilemapTargets(i);
            m_OnClose?.Invoke();
            GUIUtility.ExitGUI();
        }
    }
}
