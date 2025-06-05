using System;

using UnityEditor.VersionControl;

using Codice;
using Codice.Client.Commands.Tree;
using Codice.Client.Commands.WkTree;
using Codice.CM.Common;
using PlasticGui;
using Unity.PlasticSCM.Editor.AssetsOverlays.Cache;
using Unity.PlasticSCM.Editor.AssetsOverlays;
using Unity.PlasticSCM.Editor.AssetUtils;

namespace Unity.PlasticSCM.Editor.AssetMenu
{
    [Flags]
    internal enum AssetMenuOperations : byte
    {
        None =                   0,
        Checkout =               1 << 0,
        Diff =                   1 << 1,
        History =                1 << 2,
        Add =                    1 << 3,
        Checkin =                1 << 4,
        Undo =                   1 << 5,
        CopyFilePath =           1 << 6
    }

    internal class SelectedAssetGroupInfo
    {
        internal int SelectedCount;

        internal bool IsControlledSelection;
        internal bool IsCheckedInSelection;
        internal bool IsCheckedOutSelection;
        internal bool IsPrivateSelection;
        internal bool IsAddedSelection;
        internal bool IsFileSelection;
        internal bool IsMovedSelection;
        internal bool HasAnyAddedInSelection;
        internal bool HasAnyRemoteLockedInSelection;

        internal static SelectedAssetGroupInfo BuildFromAssetList(
            WorkspaceInfo wkInfo,
            AssetList assetList,
            IPlasticAPI plasticApi,
            IAssetStatusCache statusCache)
        {
            bool isCheckedInSelection = true;
            bool isControlledSelection = true;
            bool isCheckedOutSelection = true;
            bool isPrivateSelection = true;
            bool isAddedSelection = true;
            bool isFileSelection = true;
            bool isMovedSelection = true;
            bool hasAnyAddedInSelection = false;
            bool hasAnyRemoteLockedInSelection = false;

            int selectedCount = 0;

            foreach (Asset asset in assetList)
            {
                string fullPath = AssetsPath.GetFullPathUnderWorkspace.
                    ForAsset(wkInfo.ClientPath, asset.path);

                if (fullPath == null)
                    continue;

                SelectedAssetGroupInfo singleFileGroupInfo = BuildFromSingleFile(
                    fullPath, asset.isFolder, wkInfo, plasticApi, statusCache);

                if (!singleFileGroupInfo.IsCheckedInSelection)
                    isCheckedInSelection = false;

                if (!singleFileGroupInfo.IsControlledSelection)
                    isControlledSelection = false;

                if (!singleFileGroupInfo.IsCheckedOutSelection)
                    isCheckedOutSelection = false;

                if (!singleFileGroupInfo.IsPrivateSelection)
                    isPrivateSelection = false;

                if (!singleFileGroupInfo.IsAddedSelection)
                    isAddedSelection = false;

                if (!singleFileGroupInfo.IsFileSelection)
                    isFileSelection = false;

                if (!singleFileGroupInfo.IsMovedSelection)
                    isMovedSelection = false;

                if (singleFileGroupInfo.HasAnyAddedInSelection)
                    hasAnyAddedInSelection = true;

                if (singleFileGroupInfo.HasAnyRemoteLockedInSelection)
                    hasAnyRemoteLockedInSelection = true;

                selectedCount++;
            }

            return new SelectedAssetGroupInfo()
            {
                IsCheckedInSelection = isCheckedInSelection,
                IsCheckedOutSelection = isCheckedOutSelection,
                IsControlledSelection = isControlledSelection,
                IsPrivateSelection = isPrivateSelection,
                IsAddedSelection = isAddedSelection,
                IsFileSelection = isFileSelection,
                IsMovedSelection = isMovedSelection,
                HasAnyAddedInSelection = hasAnyAddedInSelection,
                HasAnyRemoteLockedInSelection = hasAnyRemoteLockedInSelection,
                SelectedCount = selectedCount,
            };
        }

        internal static SelectedAssetGroupInfo BuildFromSingleFile(
            string fullPath,
            bool isDirectory,
            WorkspaceInfo wkInfo,
            IPlasticAPI plasticApi,
            IAssetStatusCache statusCache)
        {
            bool isCheckedInSelection = true;
            bool isControlledSelection = true;
            bool isCheckedOutSelection = true;
            bool isPrivateSelection = true;
            bool isAddedSelection = true;
            bool isFileSelection = true;
            bool isMovedSelection = false;
            bool hasAnyAddedInSelection = false;
            bool hasAnyRemoteLockedInSelection = false;

            WorkspaceTreeNode wkTreeNode = plasticApi.
                GetWorkspaceTreeNode(fullPath);

            if (isDirectory)
                isFileSelection = false;

            if (CheckWorkspaceTreeNodeStatus.IsPrivate(wkTreeNode))
                isControlledSelection = false;
            else
                isPrivateSelection = false;

            if (CheckWorkspaceTreeNodeStatus.IsCheckedOut(wkTreeNode))
                isCheckedInSelection = false;
            else
                isCheckedOutSelection = false;

            if (CheckWorkspaceTreeNodeStatus.IsAdded(wkTreeNode))
                hasAnyAddedInSelection = true;
            else
                isAddedSelection = false;

            if (wkTreeNode != null)
            {
                TreeChangedNode treeChangedNode = plasticApi.
                    GetChangedNodeForPath(wkInfo, fullPath);

                if (treeChangedNode != null)
                    isMovedSelection = treeChangedNode.MovedDst;
            }

            AssetStatus assetStatus = statusCache.GetStatus(fullPath);

            if (ClassifyAssetStatus.IsLockedRemote(assetStatus))
                hasAnyRemoteLockedInSelection = true;

            return new SelectedAssetGroupInfo()
            {
                IsCheckedInSelection = isCheckedInSelection,
                IsCheckedOutSelection = isCheckedOutSelection,
                IsControlledSelection = isControlledSelection,
                IsPrivateSelection = isPrivateSelection,
                IsAddedSelection = isAddedSelection,
                IsFileSelection = isFileSelection,
                IsMovedSelection = isMovedSelection,
                HasAnyAddedInSelection = hasAnyAddedInSelection,
                HasAnyRemoteLockedInSelection = hasAnyRemoteLockedInSelection,
                SelectedCount = 1,
            };
        }
    }

    internal interface IAssetMenuVCSOperations
    {
        void ShowPendingChanges();
        void Add();
        void Checkout();
        void Checkin();
        void Undo();
        void ShowDiff();
        void ShowHistory();
    }

    internal interface IAssetMenuCopyPathOperation
    {
        void CopyFilePath(bool relativePath);
    }

    internal static class AssetMenuUpdater
    {
        internal static AssetMenuOperations GetAvailableMenuOperations(
            SelectedAssetGroupInfo info)
        {
            AssetMenuOperations result = AssetMenuOperations.None;

            if (info.SelectedCount == 0)
            {
                return result;
            }

            result |= AssetMenuOperations.CopyFilePath;

            if (info.IsControlledSelection &&
                info.IsCheckedInSelection &&
                info.IsFileSelection &&
                !info.HasAnyRemoteLockedInSelection)
            {
                result |= AssetMenuOperations.Checkout;
            }

            if (info.IsFileSelection &&
                info.IsPrivateSelection)
            {
                result |= AssetMenuOperations.Add;
            }

            if (info.IsFileSelection &&
                info.IsControlledSelection &&
                info.IsCheckedOutSelection)
            {
                result |= AssetMenuOperations.Checkin;
            }

            if ((info.IsControlledSelection &&
                 info.IsMovedSelection) ||
                (info.IsFileSelection &&
                info.IsControlledSelection &&
                info.IsCheckedOutSelection))
            {
                result |= AssetMenuOperations.Undo;
            }

            if (info.SelectedCount == 1 &&
                info.IsControlledSelection &&
                !info.HasAnyAddedInSelection &&
                info.IsFileSelection)
            {
                result |= AssetMenuOperations.Diff;
            }

            if (info.SelectedCount == 1 &&
                info.IsControlledSelection &&
                !info.HasAnyAddedInSelection)
            {
                result |= AssetMenuOperations.History;
            }

            return result;
        }
    }
}
