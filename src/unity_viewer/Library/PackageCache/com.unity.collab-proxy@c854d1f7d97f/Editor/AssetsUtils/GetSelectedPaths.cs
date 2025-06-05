using System.Collections.Generic;
using System.IO;

using Codice.CM.Common;
using PlasticGui;
using Unity.PlasticSCM.Editor.AssetMenu;
using Unity.PlasticSCM.Editor.AssetsOverlays.Cache;
using UnityEditor.VersionControl;

namespace Unity.PlasticSCM.Editor.AssetUtils
{
    internal static class GetSelectedPaths
    {
        internal static List<string> ForOperation(
            WorkspaceInfo wkInfo,
            IEnumerable<Asset> assetList,
            IPlasticAPI plasticApi,
            IAssetStatusCache assetStatusCache,
            AssetMenuOperations operation,
            bool includeMetaFiles = true)
        {
            List<string> selectedPaths = AssetsSelection.
                GetSelectedPaths(wkInfo.ClientPath, assetList);

            List<string> result = new List<string>(selectedPaths);

            if (!includeMetaFiles)
                return result;

            foreach (string path in selectedPaths)
            {
                if (MetaPath.IsMetaPath(path))
                    continue;

                string metaPath = MetaPath.GetMetaPath(path);

                if (!File.Exists(metaPath))
                    continue;

                if (result.Contains(metaPath))
                    continue;

                if (!IsApplicableForOperation(
                        metaPath, false, wkInfo, plasticApi, assetStatusCache, operation))
                    continue;

                result.Add(metaPath);
            }

            return result;
        }

        static bool IsApplicableForOperation(
            string path,
            bool isDirectory,
            WorkspaceInfo wkInfo,
            IPlasticAPI plasticApi,
            IAssetStatusCache assetStatusCache,
            AssetMenuOperations operation)
        {
            SelectedAssetGroupInfo info = SelectedAssetGroupInfo.BuildFromSingleFile(
                path, isDirectory, wkInfo, plasticApi, assetStatusCache);

            return AssetMenuUpdater.GetAvailableMenuOperations(info).HasFlag(operation);
        }
    }
}
