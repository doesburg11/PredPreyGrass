using System;
using System.Collections.Generic;
using UnityEngine;

using Object = UnityEngine.Object;

namespace UnityEditor.U2D.Animation.Upgrading
{
    internal static class AssetUpgrader
    {
        static Dictionary<UpgradeMode, BaseUpgrader> s_Upgraders = new Dictionary<UpgradeMode, BaseUpgrader>()
        {
            { UpgradeMode.AnimationClip, new AnimClipUpgrader() },
            { UpgradeMode.SpriteLibrary, new SpriteLibUpgrader() }
        };

        internal static List<Object> GetAllAssetsOfType(UpgradeMode upgradeMode)
        {
            if (s_Upgraders.ContainsKey(upgradeMode))
                return s_Upgraders[upgradeMode].GetUpgradableAssets();

            return null;
        }

        internal static UpgradeReport UpgradeSelection(UpgradeMode upgradeMode, List<ObjectIndexPair> selection)
        {
            UpgradeReport report = default;
            try
            {
                if (s_Upgraders.ContainsKey(upgradeMode))
                    report =  s_Upgraders[upgradeMode].UpgradeSelection(selection);
            }
            catch
            {
                EditorUtility.ClearProgressBar();
                throw;
            }

            return report;
        }
    }
}