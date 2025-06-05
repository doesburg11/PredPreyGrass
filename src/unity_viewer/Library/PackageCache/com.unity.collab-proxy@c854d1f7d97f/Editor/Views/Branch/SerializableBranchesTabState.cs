using System;

namespace Unity.PlasticSCM.Editor.Views.Branches
{
    [Serializable]
    internal class SerializableBranchesTabState
    {
        internal bool ShowHiddenBranches;

        internal bool IsInitialized { get; private set; }

        internal SerializableBranchesTabState(bool showHiddenBranches)
        {
            ShowHiddenBranches = showHiddenBranches;

            IsInitialized = true;
        }
    }
}
