using UnityEngine;

namespace UnityEditor.Tilemaps
{
    internal class TilePaletteActiveTargetsProperties
    {
        public enum PrefabEditModeSettings
        {
            EnableDialog = 0,
            EditInPrefabMode = 1,
            EditInScene = 2
        }

        public static readonly string targetEditModeDialogTitle = L10n.Tr("Open in Prefab Mode");
        public static readonly string targetEditModeDialogMessage = L10n.Tr("Editing Tilemaps in Prefabs will have better performance if edited in Prefab Mode. Do you want to open it in Prefab Mode or edit it in the Scene?");
        public static readonly string targetEditModeDialogYes = L10n.Tr("Prefab Mode");
        public static readonly string targetEditModeDialogChange = L10n.Tr("Preferences");
        public static readonly string targetEditModeDialogNo = L10n.Tr("Scene");

        public static readonly string targetEditModeEditorPref = "TilePalette.TargetEditMode";
        public static readonly string targetEditModeLookup = "Target Edit Mode";
        public static readonly string tilePalettePreferencesLookup = "Tile Palette";

        public static readonly GUIContent targetEditModeDialogLabel = EditorGUIUtility.TrTextContent(targetEditModeLookup, "Controls the behaviour of editing a Prefab Instance when one is selected as the Active Target in the Tile Palette");
    }
}
