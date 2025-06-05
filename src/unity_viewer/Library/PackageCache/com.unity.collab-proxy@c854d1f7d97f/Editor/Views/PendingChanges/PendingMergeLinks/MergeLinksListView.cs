using System.Collections.Generic;

using UnityEditor.IMGUI.Controls;
using UnityEngine;

using Codice.Client.Common;
using Codice.CM.Common;
using Codice.CM.Common.Merge;
using Codice.CM.Common.Mount;
using PlasticGui.WorkspaceWindow.PendingChanges;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.Tree;

namespace Unity.PlasticSCM.Editor.Views.PendingChanges.PendingMergeLinks
{
    internal class MergeLinksListView : PlasticTreeView
    {
        internal float DesiredHeight
        {
            get
            {
                return rowHeight * (mMergeLinks.Count + 1);
            }
        }

        protected override IList<TreeViewItem> BuildRows(TreeViewItem rootItem)
        {
            RegenerateRows(mMergeLinks, rootItem, mRows);
            return mRows;
        }

        internal void BuildModel(
            IDictionary<MountPoint, IList<PendingMergeLink>> pendingMergeLinks)
        {
            mMergeLinks = BuildMountPendingMergeLink(pendingMergeLinks);
        }

        static void RegenerateRows(
            List<MountPendingMergeLink> mergeLinks,
            TreeViewItem rootItem,
            List<TreeViewItem> rows)
        {
            ClearRows(rootItem, rows);

            if (mergeLinks.Count == 0)
                return;

            for (int i = 0; i < mergeLinks.Count; i++)
            {
                MergeLinkListViewItem mergeLinkListViewItem =
                    new MergeLinkListViewItem(i + 1, mergeLinks[i]);

                rootItem.AddChild(mergeLinkListViewItem);
                rows.Add(mergeLinkListViewItem);
            }
        }

        static void ClearRows(
            TreeViewItem rootItem,
            List<TreeViewItem> rows)
        {
            if (rootItem.hasChildren)
                rootItem.children.Clear();

            rows.Clear();
        }

        static List<MountPendingMergeLink> BuildMountPendingMergeLink(
            IDictionary<MountPoint, IList<PendingMergeLink>> pendingMergeLinks)
        {
            List<MountPendingMergeLink> result = new List<MountPendingMergeLink>();

            if (pendingMergeLinks == null)
                return result;

            foreach (KeyValuePair<MountPoint, IList<PendingMergeLink>> mountLink
                in pendingMergeLinks)
            {
                result.AddRange(BuildMountPendingMergeLinks(
                    mountLink.Key, mountLink.Value));
            }

            return result;
        }

        static List<MountPendingMergeLink> BuildMountPendingMergeLinks(
            MountPoint mount, IList<PendingMergeLink> links)
        {
            List<MountPendingMergeLink> result = new List<MountPendingMergeLink>();

            RepositoryInfo repInfo = RepositorySpecResolverProvider.Get().
                GetRepInfo(mount.RepSpec);

            foreach (PendingMergeLink link in links)
                result.Add(new MountPendingMergeLink(repInfo.GetRepSpec(), link));

            return result;
        }

        List<MountPendingMergeLink> mMergeLinks = new List<MountPendingMergeLink>();
    }
}
