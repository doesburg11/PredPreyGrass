using System;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    internal class TilePaletteActiveTargetsDropdownMenu : IGenericMenu
    {
        private const float k_ActiveTargetDropdownWidth = 200f;

        private GridPaintTargetsDropdown m_Dropdown;
        private TilePaletteCreateTargetsDropdownMenu m_Menu;

        public TilePaletteActiveTargetsDropdownMenu()
        {
            var index =
                GridPaintingState.validTargets != null && GridPaintingState.scenePaintTarget != null
                ? Array.IndexOf(GridPaintingState.validTargets, GridPaintingState.scenePaintTarget)
                : -1;
            var menuData = new GridPaintTargetsDropdown.MenuItemProvider();
            m_Dropdown = new GridPaintTargetsDropdown(menuData, index, null, SelectTarget, HoverTarget, k_ActiveTargetDropdownWidth);
        }

        private void OnClose()
        {
            PopupWindow.Show(default, m_Dropdown);
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

        private static void SelectTarget(int i, object o)
        {
            if ((GridPaintingState.validTargets == null && i > 0)
                || (GridPaintingState.validTargets != null && i >= GridPaintingState.validTargets.Length))
            {
                return;
            }

            var obj = o as GameObject;
            var isPrefabInstance = TilePalettePrefabUtility.IsObjectPrefabInstance(obj);
            if (isPrefabInstance)
            {
                var editMode = (TilePaletteActiveTargetsProperties.PrefabEditModeSettings)EditorPrefs.GetInt(TilePaletteActiveTargetsProperties.targetEditModeEditorPref, 0);
                switch (editMode)
                {
                    case TilePaletteActiveTargetsProperties.PrefabEditModeSettings.EnableDialog:
                    {
                        var option = EditorUtility.DisplayDialogComplex(TilePaletteActiveTargetsProperties.targetEditModeDialogTitle
                            , TilePaletteActiveTargetsProperties.targetEditModeDialogMessage
                            , TilePaletteActiveTargetsProperties.targetEditModeDialogYes
                            , TilePaletteActiveTargetsProperties.targetEditModeDialogNo
                            , TilePaletteActiveTargetsProperties.targetEditModeDialogChange);
                        switch (option)
                        {
                            case 0:
                                TilePalettePrefabUtility.GoToPrefabMode(obj);
                                return;
                            case 1:
                                // Do nothing here for "No"
                                break;
                            case 2:
                                var settingsWindow = SettingsWindow.Show(SettingsScope.User);
                                settingsWindow.FilterProviders(TilePaletteActiveTargetsProperties.targetEditModeLookup);
                                break;
                        }
                    }
                    break;
                    case TilePaletteActiveTargetsProperties.PrefabEditModeSettings.EditInPrefabMode:
                        TilePalettePrefabUtility.GoToPrefabMode(obj);
                        return;
                }
            }

            GridPaintingState.scenePaintTarget = obj;
        }

        private void HoverTarget(int index, Rect itemRect)
        {
            var targets = GridPaintingState.validTargets;
            var count = 0;
            if (targets != null)
                count = targets.Length;

            if (index < count)
            {
                if (m_Menu != null)
                {
                    m_Menu.Close();
                    m_Menu = null;
                }
                return;
            }

            if (!GridPaintCreateTargetsDropdown.IsOpen)
            {
                m_Menu = new TilePaletteCreateTargetsDropdownMenu(OnClose);

                var popupRect = itemRect;
                popupRect.x += itemRect.width;
                popupRect.y -= itemRect.height;

                m_Menu.DropDown(popupRect);
            }
        }
    }
}
