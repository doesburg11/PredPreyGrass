using System;
using System.Collections.Generic;
using UnityEngine;

namespace UnityEditor.Tilemaps
{
    internal class GridPalettes : ScriptableSingleton<GridPalettes>
    {
        private List<GameObject> m_PalettesCache;

        internal static Action palettesChanged;

        internal static List<GameObject> palettes
        {
            get
            {
                if (instance.m_PalettesCache == null
                    || (instance.m_PalettesCache.Count > 0 && instance.m_PalettesCache[0] == null))
                {
                    instance.RefreshPalettesCache();
                }
                return instance.m_PalettesCache;
            }
        }

        private void OnEnable()
        {
            CleanCache();
        }

        private void OnDisable()
        {
            CleanCache();
        }

        private void RefreshPalettesCache()
        {
            if (m_PalettesCache == null)
                m_PalettesCache = new List<GameObject>();
            m_PalettesCache.Clear();

            string[] guids = AssetDatabase.FindAssets("t:GridPalette");
            foreach (string guid in guids)
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                GridPalette paletteAsset = AssetDatabase.LoadAssetAtPath(path, typeof(GridPalette)) as GridPalette;
                if (paletteAsset != null)
                {
                    string assetPath = AssetDatabase.GetAssetPath(paletteAsset);
                    GameObject palette = AssetDatabase.LoadMainAssetAtPath(assetPath) as GameObject;
                    if (palette != null)
                    {
                        m_PalettesCache.Add(palette);
                    }
                }
            }
            m_PalettesCache.Sort((x, y) => String.Compare(x.name, y.name, StringComparison.OrdinalIgnoreCase));

            palettesChanged?.Invoke();
        }

        private class AssetProcessor : AssetPostprocessor
        {
            public override int GetPostprocessOrder()
            {
                return 1;
            }

            private static void OnPostprocessAllAssets(string[] importedAssets, string[] deletedAssets, string[] movedAssets, string[] movedFromPath)
            {
                if (!GridPaintingState.savingPalette)
                {
                    CleanCache();
                    palettesChanged?.Invoke();
                }
            }
        }

        internal static void CleanCache()
        {
            instance.m_PalettesCache = null;
        }
    }
}
