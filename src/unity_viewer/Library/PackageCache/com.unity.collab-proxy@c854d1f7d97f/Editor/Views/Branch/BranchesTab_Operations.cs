using Codice.CM.Common;
using GluonGui.WorkspaceWindow.Views.WorkspaceExplorer.Explorer;
using PlasticGui.WorkspaceWindow.QueryViews.Branches;
using Unity.PlasticSCM.Editor.AssetUtils;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.Views.Branches.Dialogs;

namespace Unity.PlasticSCM.Editor.Views.Branches
{
    internal partial class BranchesTab
    {
        void SwitchToBranchForMode()
        {
            bool isCancelled;
            mSaveAssets.UnderWorkspaceWithConfirmation(
                mWkInfo.ClientPath, mWorkspaceOperationsMonitor,
                out isCancelled);

            if (isCancelled)
                return;

            if (mIsGluonMode)
            {
                SwitchToBranchForGluon();
                return;
            }

            SwitchToBranchForDeveloper();
        }

        void SwitchToBranchForDeveloper()
        {
            RepositorySpec repSpec = BranchesSelection.GetSelectedRepository(mBranchesListView);
            BranchInfo branchInfo = BranchesSelection.GetSelectedBranch(mBranchesListView);

            mBranchOperations.SwitchToBranch(
                repSpec,
                branchInfo,
                RefreshAsset.BeforeLongAssetOperation,
                items => RefreshAsset.AfterLongAssetOperation(
                    ProjectPackages.ShouldBeResolvedFromUpdateReport(mWkInfo, items)));
        }

        void SwitchToBranchForGluon()
        {
            BranchInfo branchInfo = BranchesSelection.GetSelectedBranch(mBranchesListView);

            new SwitchToUIOperation().SwitchToBranch(
                mWkInfo,
                branchInfo,
                mViewHost,
                mGluonNewIncomingChangesUpdater,
                new UnityPlasticGuiMessage(),
                mProgressControls,
                mWorkspaceWindow.GluonProgressOperationHandler,
                mGluonUpdateReport,
                mWorkspaceWindow,
                mShelvePendingChangesQuestionerBuilder,
                mShelvedChangesUpdater,
                mEnableSwitchAndShelveFeatureDialog,
                RefreshAsset.BeforeLongAssetOperation,
                items => RefreshAsset.AfterLongAssetOperation(
                    ProjectPackages.ShouldBeResolvedFromPaths(mWkInfo, items)));
        }

        void CreateBranchForMode()
        {
            if (mIsGluonMode)
            {
                CreateBranchForGluon();
                return;
            }

            CreateBranchForDeveloper();
        }

        void CreateBranchForDeveloper()
        {
            RepositorySpec repSpec = BranchesSelection.GetSelectedRepository(mBranchesListView);
            BranchInfo branchInfo = BranchesSelection.GetSelectedBranch(mBranchesListView);

            BranchCreationData branchCreationData = CreateBranchDialog.CreateBranchFromLastParentBranchChangeset(
                mParentWindow,
                repSpec,
                branchInfo);

            mBranchOperations.CreateBranch(
                branchCreationData,
                RefreshAsset.BeforeLongAssetOperation,
                items => RefreshAsset.AfterLongAssetOperation(
                    ProjectPackages.ShouldBeResolvedFromUpdateReport(mWkInfo, items)));
        }

        void CreateBranchForGluon()
        {
            RepositorySpec repSpec = BranchesSelection.GetSelectedRepository(mBranchesListView);
            BranchInfo branchInfo = BranchesSelection.GetSelectedBranch(mBranchesListView);

            BranchCreationData branchCreationData = CreateBranchDialog.CreateBranchFromLastParentBranchChangeset(
                mParentWindow,
                repSpec,
                branchInfo);

            CreateBranchOperation.CreateBranch(
                mWkInfo,
                branchCreationData,
                mViewHost,
                mGluonNewIncomingChangesUpdater,
                new UnityPlasticGuiMessage(),
                mProgressControls,
                mWorkspaceWindow.GluonProgressOperationHandler,
                mGluonUpdateReport,
                mWorkspaceWindow,
                mShelvePendingChangesQuestionerBuilder,
                mShelvedChangesUpdater,
                mEnableSwitchAndShelveFeatureDialog,
                RefreshAsset.BeforeLongAssetOperation,
                items => RefreshAsset.AfterLongAssetOperation(
                    ProjectPackages.ShouldBeResolvedFromPaths(mWkInfo, items)));
        }
    }
}
