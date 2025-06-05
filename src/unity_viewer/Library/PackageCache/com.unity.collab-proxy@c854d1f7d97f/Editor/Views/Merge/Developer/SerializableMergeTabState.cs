using System;

using UnityEngine;

using Codice.CM.Common;
using PlasticGui.WorkspaceWindow.Merge;

namespace Unity.PlasticSCM.Editor.Views.Merge.Developer
{
    [Serializable]
    internal class SerializableMergeTabState
    {
        internal RepositorySpec RepSpec { get; private set; }
        internal EnumMergeType MergeType { get; private set; }
        internal ShowIncomingChangesFrom From { get; private set; }
        internal bool IsIncomingMerge { get; private set; }
        internal bool IsMergeFinished { get; set; }

        internal bool IsInitialized { get; private set; }

        internal SerializableMergeTabState(
            RepositorySpec repSpec,
            ObjectInfo objectInfo,
            ObjectInfo ancestorObjectInfo,
            EnumMergeType mergeType,
            ShowIncomingChangesFrom from,
            bool isIncomingMerge,
            bool isMergeFinished)
        {
            RepSpec = repSpec;

            SetObjectInfo(objectInfo);
            SetAncestorObjectInfo(ancestorObjectInfo);

            MergeType = mergeType;
            From = from;
            IsIncomingMerge = isIncomingMerge;
            IsMergeFinished = isMergeFinished;

            IsInitialized = true;
        }

        internal ObjectInfo GetObjectInfo()
        {
            if (mBranchInfo != null && mBranchInfo.Id != -1)
                return mBranchInfo;

            if (mChangesetInfo != null && mChangesetInfo.Id != -1)
                return mChangesetInfo;

            if (mLabelInfo != null && mLabelInfo.Id != -1)
                return mLabelInfo;

            return null;
        }

        internal ObjectInfo GetAncestorObjectInfo()
        {
            if (mAncestorBranchInfo != null && mAncestorBranchInfo.Id != -1)
                return mAncestorBranchInfo;

            if (mAncestorChangesetInfo != null && mAncestorChangesetInfo.Id != -1)
                return mAncestorChangesetInfo;

            if (mAncestorLabelInfo != null && mAncestorLabelInfo.Id != -1)
                return mAncestorLabelInfo;

            return null;
        }

        void SetObjectInfo(ObjectInfo objectInfo)
        {
            if (objectInfo is BranchInfo)
            {
                mBranchInfo = (BranchInfo)objectInfo;
                return;
            }

            if (objectInfo is ChangesetInfo)
            {
                mChangesetInfo = (ChangesetInfo)objectInfo;
                return;
            }

            if (objectInfo is MarkerInfo)
            {
                mLabelInfo = (MarkerInfo)objectInfo;
                return;
            }
        }

        void SetAncestorObjectInfo(ObjectInfo objectInfo)
        {
            if (objectInfo is BranchInfo)
            {
                mAncestorBranchInfo = (BranchInfo)objectInfo;
                return;
            }

            if (objectInfo is ChangesetInfo)
            {
                mAncestorChangesetInfo = (ChangesetInfo)objectInfo;
                return;
            }

            if (objectInfo is MarkerInfo)
            {
                mAncestorLabelInfo = (MarkerInfo)objectInfo;
                return;
            }
        }

        [SerializeField]
        BranchInfo mBranchInfo;
        [SerializeField]
        ChangesetInfo mChangesetInfo;
        [SerializeField]
        MarkerInfo mLabelInfo;

        [SerializeField]
        BranchInfo mAncestorBranchInfo;
        [SerializeField]
        ChangesetInfo mAncestorChangesetInfo;
        [SerializeField]
        MarkerInfo mAncestorLabelInfo;
    }
}
