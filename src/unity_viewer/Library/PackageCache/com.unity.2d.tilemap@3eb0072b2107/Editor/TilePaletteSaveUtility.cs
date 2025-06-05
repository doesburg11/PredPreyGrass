using System;
using UnityEngine;

namespace UnityEditor.Tilemaps
{
    internal class TilePaletteSaveUtility
    {
        private static class Styles
        {
            public static readonly string invalidFolderTitle = L10n.Tr("Cannot save to an invalid folder");
            public static readonly string invalidFolderContent = L10n.Tr("You cannot save to an invalid folder.");
            public static readonly string nonAssetFolderTitle = L10n.Tr("Cannot save to a non-asset folder");
            public static readonly string nonAssetFolderContent = L10n.Tr("You cannot save to a non-asset folder.");
            public static readonly string readOnlyFolderTitle = L10n.Tr("Cannot save to a read-only path");
            public static readonly string readOnlyFolderContent = L10n.Tr("You cannot save to a read-only path.");
            public static readonly string ok = L10n.Tr("OK");
        }

        private class TilePaletteSaveScope : IDisposable
        {
            private GameObject m_GameObject;

            public TilePaletteSaveScope(GameObject paletteInstance)
            {
                m_GameObject = paletteInstance;
                if (m_GameObject != null)
                {
                    GridPaintingState.savingPalette = true;
                    SetHideFlagsRecursively(paletteInstance, HideFlags.HideInHierarchy);
                    var renderers = paletteInstance.GetComponentsInChildren<Renderer>();
                    foreach (var renderer in renderers)
                        renderer.gameObject.layer = 0;
                }
            }

            public void Dispose()
            {
                if (m_GameObject != null)
                {
                    SetHideFlagsRecursively(m_GameObject, HideFlags.HideAndDontSave);
                    GridPaintingState.savingPalette = false;
                }
            }

            private void SetHideFlagsRecursively(GameObject root, HideFlags flags)
            {
                root.hideFlags = flags;
                for (int i = 0; i < root.transform.childCount; i++)
                    SetHideFlagsRecursively(root.transform.GetChild(i).gameObject, flags);
            }
        }

        public static bool SaveTilePalette(GameObject originalPalette, GameObject paletteInstance)
        {
            var path = PrefabUtility.GetPrefabAssetPathOfNearestInstanceRoot(originalPalette);
            if (path == null)
                return false;

            using (new TilePaletteSaveScope(paletteInstance))
            {
                PrefabUtility.SaveAsPrefabAssetAndConnect(paletteInstance, path, InteractionMode.AutomatedAction);
            }
            return true;
        }

        public static bool ValidateSaveFolder(string folderPath)
        {
            if (string.IsNullOrEmpty(folderPath))
            {
                EditorUtility.DisplayDialog(Styles.invalidFolderTitle, Styles.invalidFolderContent, Styles.ok);
                return false;
            }
            if (!AssetDatabase.TryGetAssetFolderInfo(folderPath, out bool rootFolder, out bool immutable))
            {
                EditorUtility.DisplayDialog(Styles.nonAssetFolderTitle, Styles.nonAssetFolderContent, Styles.ok);
                return false;
            }
            if (immutable)
            {
                EditorUtility.DisplayDialog(Styles.readOnlyFolderTitle, Styles.readOnlyFolderContent, Styles.ok);
                return false;
            }
            return true;
        }
    }
}
