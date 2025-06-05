using System.Collections.Generic;

using UnityEditor;

using Codice.CM.Common;
using PlasticGui;
using PlasticGui.WorkspaceWindow;
using Unity.PlasticSCM.Editor.AssetsOverlays.Cache;
using Unity.PlasticSCM.Editor.AssetUtils;

namespace Unity.PlasticSCM.Editor.AssetMenu
{
    internal class AssetCopyPathOperation : IAssetMenuCopyPathOperation
    {
        internal AssetCopyPathOperation(
            WorkspaceInfo wkInfo,
            IPlasticAPI plasticApi,
            IAssetStatusCache assetStatusCache,
            AssetVcsOperations.IAssetSelection assetSelection)
        {
            mWkInfo = wkInfo;
            mPlasticAPI = plasticApi;
            mAssetStatusCache = assetStatusCache;
            mAssetSelection = assetSelection;
        }

        void IAssetMenuCopyPathOperation.CopyFilePath(bool relativePath)
        {
            List<string> selectedPaths = GetSelectedPaths.ForOperation(
                mWkInfo,
                mAssetSelection.GetSelectedAssets(),
                mPlasticAPI,
                mAssetStatusCache,
                AssetMenuOperations.CopyFilePath,
                includeMetaFiles: false);

            EditorGUIUtility.systemCopyBuffer = GetFilePathList.FromSelectedPaths(
                selectedPaths,
                relativePath,
                mWkInfo.ClientPath);
        }

        readonly WorkspaceInfo mWkInfo;
        readonly IPlasticAPI mPlasticAPI;
        readonly IAssetStatusCache mAssetStatusCache;
        readonly AssetVcsOperations.IAssetSelection mAssetSelection;
    }
}
