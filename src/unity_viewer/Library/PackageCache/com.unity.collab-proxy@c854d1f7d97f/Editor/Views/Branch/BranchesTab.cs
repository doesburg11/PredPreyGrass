using System;
using System.Collections.Generic;

using UnityEditor;
using UnityEditor.IMGUI.Controls;
using UnityEngine;

using Codice.Client.Common.EventTracking;
using Codice.Client.Common.Threading;
using Codice.CM.Common;
using GluonGui;
using PlasticGui;
using PlasticGui.WorkspaceWindow;
using PlasticGui.WorkspaceWindow.CodeReview;
using PlasticGui.WorkspaceWindow.QueryViews;
using PlasticGui.WorkspaceWindow.QueryViews.Branches;
using PlasticGui.WorkspaceWindow.Update;
using Unity.PlasticSCM.Editor.AssetUtils;
using Unity.PlasticSCM.Editor.AssetUtils.Processor;
using Unity.PlasticSCM.Editor.Tool;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.Progress;
using Unity.PlasticSCM.Editor.UI.Tree;
using Unity.PlasticSCM.Editor.Views.Branches.Dialogs;
using Unity.PlasticSCM.Editor.Views.Changesets;

using GluonNewIncomingChangesUpdater = PlasticGui.Gluon.WorkspaceWindow.NewIncomingChangesUpdater;
using IGluonUpdateReport = PlasticGui.Gluon.IUpdateReport;

namespace Unity.PlasticSCM.Editor.Views.Branches
{
    internal partial class BranchesTab :
        IRefreshableView,
        IQueryRefreshableView,
        IBranchMenuOperations,
        ILaunchCodeReviewWindow
    {
        internal bool ShowHiddenBranchesForTesting { set { mShowHiddenBranches = value; } }
        internal string EmptyStateMessage { get { return mEmptyStateData.Content.text; } }
        internal BranchesListView Table { get { return mBranchesListView; } }
        internal IBranchMenuOperations Operations { get { return this; } }

        internal BranchesTab(
            WorkspaceInfo wkInfo,
            WorkspaceWindow workspaceWindow,
            IViewSwitcher viewSwitcher,
            IMergeViewLauncher mergeViewLauncher,
            ViewHost viewHost,
            IUpdateReport updateReport,
            IGluonUpdateReport gluonUpdateReport,
            NewIncomingChangesUpdater developerNewIncomingChangesUpdater,
            GluonNewIncomingChangesUpdater gluonNewIncomingChangesUpdater,
            IShelvedChangesUpdater shelvedChangesUpdater,
            LaunchTool.IShowDownloadPlasticExeWindow showDownloadPlasticExeWindow,
            LaunchTool.IProcessExecutor processExecutor,
            WorkspaceOperationsMonitor workspaceOperationsMonitor,
            ISaveAssets saveAssets,
            EditorWindow parentWindow,
            bool isGluonMode,
            bool showHiddenBranches)
        {
            mWkInfo = wkInfo;
            mParentWindow = parentWindow;
            mGluonUpdateReport = gluonUpdateReport;
            mViewHost = viewHost;
            mWorkspaceWindow = workspaceWindow;
            mIsGluonMode = isGluonMode;
            mShowHiddenBranches = showHiddenBranches;
            mProgressControls = new ProgressControlsForViews();

            mDeveloperNewIncomingChangesUpdater = developerNewIncomingChangesUpdater;
            mGluonNewIncomingChangesUpdater = gluonNewIncomingChangesUpdater;
            mShelvedChangesUpdater = shelvedChangesUpdater;
            mShowDownloadPlasticExeWindow = showDownloadPlasticExeWindow;
            mProcessExecutor = processExecutor;
            mWorkspaceOperationsMonitor = workspaceOperationsMonitor;
            mSaveAssets = saveAssets;
            mShelvePendingChangesQuestionerBuilder =
                new ShelvePendingChangesQuestionerBuilder(parentWindow);
            mEnableSwitchAndShelveFeatureDialog = new EnableSwitchAndShelveFeature(
                PlasticGui.Plastic.API.GetRepositorySpec(mWkInfo),
                mParentWindow);

            BuildComponents(
                wkInfo,
                workspaceWindow,
                viewSwitcher,
                mergeViewLauncher,
                updateReport,
                developerNewIncomingChangesUpdater,
                shelvedChangesUpdater,
                mShelvePendingChangesQuestionerBuilder,
                mEnableSwitchAndShelveFeatureDialog,
                parentWindow);

            ((IRefreshableView)this).Refresh();
        }

        internal void OnEnable()
        {
            mSearchField.downOrUpArrowKeyPressed +=
                SearchField_OnDownOrUpArrowKeyPressed;
        }

        internal void OnDisable()
        {
            mSearchField.downOrUpArrowKeyPressed -=
                SearchField_OnDownOrUpArrowKeyPressed;

            TreeHeaderSettings.Save(
                mBranchesListView.multiColumnHeader.state,
                UnityConstants.BRANCHES_TABLE_SETTINGS_NAME);
        }

        internal SerializableBranchesTabState GetSerializableState()
        {
            return new SerializableBranchesTabState(mShowHiddenBranches);
        }

        internal void Update()
        {
            mProgressControls.UpdateProgress(mParentWindow);
        }

        internal void OnGUI()
        {
            DoActionsToolbar(mProgressControls);

            DoBranchesArea(
                mBranchesListView,
                mEmptyStateData,
                mProgressControls.IsOperationRunning(),
                mShowHiddenBranches,
                HasFiltersApplied(mDateFilter, mBranchesListView),
                mParentWindow.Repaint);
        }

        internal void DrawSearchFieldForTab()
        {
            DrawSearchField.For(
                mSearchField,
                mBranchesListView,
                UnityConstants.SEARCH_FIELD_WIDTH);
        }

        internal void DrawShowHiddenBranchesButton()
        {
            GUI.enabled = !mProgressControls.IsOperationRunning();

            EditorGUI.BeginChangeCheck();

            mShowHiddenBranches = GUILayout.Toggle(
                mShowHiddenBranches,
                new GUIContent(
                    mShowHiddenBranches ?
                        Images.GetUnhideIcon() :
                        Images.GetHideIcon(),
                    mShowHiddenBranches ?
                        PlasticLocalization.Name.DontShowHiddenBranchesTooltip.GetString() :
                        PlasticLocalization.Name.ShowHiddenBranchesTooltip.GetString()),
                EditorStyles.toolbarButton,
                GUILayout.Width(26));

            if (EditorGUI.EndChangeCheck())
            {
                TrackFeatureUseEvent.For(
                    BranchesSelection.GetSelectedRepository(mBranchesListView),
                    TrackFeatureUseEvent.Features.Branches.ToggleShowHiddenBranches);

                ((IRefreshableView)this).Refresh();
            }

            GUI.enabled = true;
        }

        internal void DrawDateFilter()
        {
            GUI.enabled = !mProgressControls.IsOperationRunning();

            EditorGUI.BeginChangeCheck();

            mDateFilter.FilterType = (DateFilter.Type)
                EditorGUILayout.EnumPopup(
                    mDateFilter.FilterType,
                    EditorStyles.toolbarDropDown,
                    GUILayout.Width(100));

            if (EditorGUI.EndChangeCheck())
            {
                EnumPopupSetting<DateFilter.Type>.Save(
                    mDateFilter.FilterType,
                    UnityConstants.BRANCHES_DATE_FILTER_SETTING_NAME);

                ((IRefreshableView)this).Refresh();
            }

            GUI.enabled = true;
        }

        internal void SetWorkingObjectInfo(WorkingObjectInfo homeInfo)
        {
            lock (mLock)
            {
                mLoadedBranchId = homeInfo.BranchInfo.BranchId;
            }

            mBranchesListView.SetLoadedBranchId(mLoadedBranchId);
        }

        internal void SetLaunchToolForTesting(
            LaunchTool.IShowDownloadPlasticExeWindow showDownloadPlasticExeWindow,
            LaunchTool.IProcessExecutor processExecutor)
        {
            mShowDownloadPlasticExeWindow = showDownloadPlasticExeWindow;
            mProcessExecutor = processExecutor;
        }

        void IRefreshableView.Refresh()
        {
            // VCS-1005209 - There are scenarios where the list of branches need to check for incoming changes.
            // For example, deleting the active branch will automatically switch your workspace to the parent changeset,
            // which might have incoming changes.
            if (mDeveloperNewIncomingChangesUpdater != null)
                mDeveloperNewIncomingChangesUpdater.Update(DateTime.Now);

            if (mGluonNewIncomingChangesUpdater != null)
                mGluonNewIncomingChangesUpdater.Update(DateTime.Now);

            string query = QueryConstants.BuildBranchesQuery(
                mDateFilter.GetLayoutFilter(), mShowHiddenBranches);

            FillBranches(mWkInfo, query, BranchesSelection.
                GetSelectedRepObjectInfos(mBranchesListView));
        }

        //IQueryRefreshableView
        public void RefreshAndSelect(RepObjectInfo repObj)
        {
            string query = QueryConstants.BuildBranchesQuery(
                mDateFilter.GetLayoutFilter(), mShowHiddenBranches);

            FillBranches(mWkInfo, query, new List<RepObjectInfo> { repObj });
        }

        BranchInfo IBranchMenuOperations.GetSelectedBranch()
        {
            return BranchesSelection.GetSelectedBranch(mBranchesListView);
        }

        int IBranchMenuOperations.GetSelectedBranchesCount()
        {
            return BranchesSelection.GetSelectedBranchesCount(mBranchesListView);
        }

        bool IBranchMenuOperations.AreHiddenBranchesShown()
        {
            return mShowHiddenBranches;
        }

        void IBranchMenuOperations.CreateBranch()
        {
            CreateBranchForMode();
        }

        void IBranchMenuOperations.CreateTopLevelBranch() { }

        void IBranchMenuOperations.SwitchToBranch()
        {
            SwitchToBranchForMode();
        }

        void IBranchMenuOperations.MergeBranch()
        {
            mBranchOperations.MergeBranch(
                BranchesSelection.GetSelectedRepository(mBranchesListView),
                BranchesSelection.GetSelectedBranch(mBranchesListView));
        }

        void IBranchMenuOperations.CherrypickBranch() { }

        void IBranchMenuOperations.MergeToBranch() { }

        void IBranchMenuOperations.PullBranch() { }

        void IBranchMenuOperations.PullRemoteBranch() { }

        void IBranchMenuOperations.SyncWithGit() { }

        void IBranchMenuOperations.PushBranch() { }

        void IBranchMenuOperations.DiffBranch()
        {
            LaunchDiffOperations.DiffBranch(
                mShowDownloadPlasticExeWindow,
                mProcessExecutor,
                BranchesSelection.GetSelectedRepository(mBranchesListView),
                BranchesSelection.GetSelectedBranch(mBranchesListView),
                mIsGluonMode);
        }

        void IBranchMenuOperations.DiffWithAnotherBranch() { }

        void IBranchMenuOperations.ViewChangesets() { }

        void IBranchMenuOperations.RenameBranch()
        {
            RepositorySpec repSpec = BranchesSelection.GetSelectedRepository(mBranchesListView);
            BranchInfo branchInfo = BranchesSelection.GetSelectedBranch(mBranchesListView);

            BranchRenameData branchRenameData = RenameBranchDialog.GetBranchRenameData(
                repSpec,
                branchInfo,
                mParentWindow);

            mBranchOperations.RenameBranch(branchRenameData);
        }

        void IBranchMenuOperations.HideUnhideBranch()
        {
            if (mShowHiddenBranches)
            {
                mBranchOperations.UnhideBranch(
                    BranchesSelection.GetSelectedRepositories(mBranchesListView),
                    BranchesSelection.GetSelectedBranches(mBranchesListView));
                return;
            }

            mBranchOperations.HideBranch(
                BranchesSelection.GetSelectedRepositories(mBranchesListView),
                BranchesSelection.GetSelectedBranches(mBranchesListView));
        }

        void IBranchMenuOperations.DeleteBranch()
        {
            var branchesToDelete = BranchesSelection.GetSelectedBranches(mBranchesListView);

            if (!DeleteBranchDialog.ConfirmDelete(branchesToDelete))
                return;

            mBranchOperations.DeleteBranch(
                BranchesSelection.GetSelectedRepositories(mBranchesListView),
                branchesToDelete,
                DeleteBranchOptions.IncludeChangesets,
                !mShowHiddenBranches);
        }

        void IBranchMenuOperations.CreateCodeReview()
        {
            RepositorySpec repSpec = BranchesSelection.GetSelectedRepository(mBranchesListView);
            BranchInfo branchInfo = BranchesSelection.GetSelectedBranch(mBranchesListView);

            NewCodeReviewBehavior choice = SelectNewCodeReviewBehavior.For(repSpec.Server);

            switch (choice)
            {
                case NewCodeReviewBehavior.CreateAndOpenInDesktop:
                    mBranchOperations.CreateCodeReview(repSpec, branchInfo, this);
                    break;
                case NewCodeReviewBehavior.RequestFromUnityCloud:
                    OpenRequestReviewPage.ForBranch(repSpec, branchInfo.BranchId);
                    break;
                case NewCodeReviewBehavior.Ask:
                default:
                    break;
            }
        }

        void ILaunchCodeReviewWindow.Show(
            WorkspaceInfo wkInfo,
            RepositorySpec repSpec,
            ReviewInfo reviewInfo,
            RepObjectInfo repObjectInfo,
            bool bShowReviewChangesTab)
        {
            LaunchTool.OpenCodeReview(
                mShowDownloadPlasticExeWindow,
                mProcessExecutor,
                repSpec,
                reviewInfo.Id,
                mIsGluonMode);
        }

        void IBranchMenuOperations.ViewPermissions() { }

        void SearchField_OnDownOrUpArrowKeyPressed()
        {
            mBranchesListView.SetFocusAndEnsureSelectedItem();
        }

        void OnBranchesListViewSizeChanged()
        {
            if (!mShouldScrollToSelection)
                return;

            mShouldScrollToSelection = false;
            TableViewOperations.ScrollToSelection(mBranchesListView);
        }

        void FillBranches(
            WorkspaceInfo wkInfo,
            string query,
            List<RepObjectInfo> branchesToSelect)
        {
            if (mIsRefreshing)
                return;

            mIsRefreshing = true;

            int defaultRow = TableViewOperations.
                GetFirstSelectedRow(mBranchesListView);

            ((IProgressControls)mProgressControls).ShowProgress(
                PlasticLocalization.GetString(
                    PlasticLocalization.Name.LoadingBranches));

            ViewQueryResult queryResult = null;

            IThreadWaiter waiter = ThreadWaiter.GetWaiter();
            waiter.Execute(
                /*threadOperationDelegate*/ delegate
                {
                    long loadedBranchId = GetLoadedBranchId(wkInfo);
                    lock(mLock)
                    {
                        mLoadedBranchId = loadedBranchId;
                    }

                    queryResult = new ViewQueryResult(
                        PlasticGui.Plastic.API.FindQuery(wkInfo, query));
                },
                /*afterOperationDelegate*/ delegate
                {
                    try
                    {
                        if (waiter.Exception != null)
                        {
                            ExceptionsHandler.DisplayException(waiter.Exception);
                            return;
                        }

                        UpdateBranchesList(
                            mBranchesListView,
                            queryResult,
                            mLoadedBranchId);

                        int branchesCount = GetBranchesCount(queryResult);

                        if (branchesCount == 0)
                        {
                            return;
                        }

                        BranchesSelection.SelectBranches(
                            mBranchesListView, branchesToSelect, defaultRow);
                    }
                    finally
                    {
                        ((IProgressControls)mProgressControls).HideProgress();
                        mIsRefreshing = false;
                    }
                });
        }

        static void UpdateBranchesList(
             BranchesListView branchesListView,
             ViewQueryResult queryResult,
             long loadedBranchId)
        {
            branchesListView.BuildModel(
                queryResult, loadedBranchId);

            branchesListView.Refilter();

            branchesListView.Sort();

            branchesListView.Reload();
        }

        static string GetEmptyStateMessage(
            BranchesListView branchesListView,
            bool isOperationRunning,
            bool showHiddenBranches,
            bool hasFiltersApplied)
        {
            if (isOperationRunning ||
                branchesListView.GetRows().Count > 0)
                return string.Empty;

            if (!showHiddenBranches)
                return PlasticLocalization.Name.BranchesEmptyState.GetString();

            return hasFiltersApplied ?
                PlasticLocalization.Name.HiddenBranchesEmptyState.GetString() :
                PlasticLocalization.Name.HiddenBranchesInRepositoryEmptyState.GetString();
        }

        static long GetLoadedBranchId(WorkspaceInfo wkInfo)
        {
            BranchInfo brInfo = PlasticGui.Plastic.API.GetWorkingBranch(wkInfo);

            if (brInfo != null)
                return brInfo.BranchId;

            return -1;
        }

        static int GetBranchesCount(
            ViewQueryResult queryResult)
        {
            if (queryResult == null)
                return 0;

            return queryResult.Count();
        }

        static bool HasFiltersApplied(
            DateFilter dateFilter,
            BranchesListView branchesListView)
        {
            return dateFilter.FilterType != DateFilter.Type.AllTime
                || branchesListView.searchString != string.Empty;
        }

        static void DoActionsToolbar(ProgressControlsForViews progressControls)
        {
            EditorGUILayout.BeginHorizontal(EditorStyles.toolbar);

            if (progressControls.IsOperationRunning())
            {
                DrawProgressForViews.ForIndeterminateProgress(
                    progressControls.ProgressData);
            }

            GUILayout.FlexibleSpace();

            EditorGUILayout.EndHorizontal();
        }

        static void DoBranchesArea(
            BranchesListView branchesListView,
            EmptyStateData emptyStateData,
            bool isOperationRunning,
            bool showHiddenBranches,
            bool hasFiltersApplied,
            Action repaint)
        {
            EditorGUILayout.BeginVertical();

            GUI.enabled = !isOperationRunning;

            Rect rect = GUILayoutUtility.GetRect(0, 100000, 0, 100000);

            branchesListView.OnGUI(rect);

            emptyStateData.Update(
                GetEmptyStateMessage(
                    branchesListView,
                    isOperationRunning,
                    showHiddenBranches,
                    hasFiltersApplied),
                rect, Event.current.type, repaint);

            if (!emptyStateData.IsEmpty())
                DrawTreeViewEmptyState.For(emptyStateData);

            GUI.enabled = true;

            EditorGUILayout.EndVertical();
        }

        void BuildComponents(
            WorkspaceInfo wkInfo,
            IWorkspaceWindow workspaceWindow,
            IViewSwitcher viewSwitcher,
            IMergeViewLauncher mergeViewLauncher,
            IUpdateReport updateReport,
            NewIncomingChangesUpdater developerNewIncomingChangesUpdater,
            IShelvedChangesUpdater shelvedChangesUpdater,
            IShelvePendingChangesQuestionerBuilder shelvePendingChangesQuestionerBuilder,
            SwitchAndShelve.IEnableSwitchAndShelveFeatureDialog enableSwitchAndShelveFeatureDialog,
            EditorWindow parentWindow)
        {
            mSearchField = new SearchField();
            mSearchField.downOrUpArrowKeyPressed += SearchField_OnDownOrUpArrowKeyPressed;

            DateFilter.Type dateFilterType =
                EnumPopupSetting<DateFilter.Type>.Load(
                    UnityConstants.BRANCHES_DATE_FILTER_SETTING_NAME,
                    DateFilter.Type.LastMonth);
            mDateFilter = new DateFilter(dateFilterType);

            BranchesListHeaderState headerState =
                BranchesListHeaderState.GetDefault();

            TreeHeaderSettings.Load(headerState,
                UnityConstants.BRANCHES_TABLE_SETTINGS_NAME,
                (int)BranchesListColumn.CreationDate, false);

            mBranchesListView = new BranchesListView(
                headerState,
                BranchesListHeaderState.GetColumnNames(),
                new BranchesViewMenu(this, mGluonNewIncomingChangesUpdater != null),
                sizeChangedAction: OnBranchesListViewSizeChanged,
                doubleClickAction: ((IBranchMenuOperations)this).DiffBranch);

            mBranchesListView.Reload();

            mBranchOperations = new BranchOperations(
                wkInfo,
                workspaceWindow,
                mergeViewLauncher,
                this,
                ViewType.BranchesView,
                mProgressControls,
                updateReport,
                new ContinueWithPendingChangesQuestionerBuilder(viewSwitcher, parentWindow),
                shelvePendingChangesQuestionerBuilder,
                new ApplyShelveWithConflictsQuestionerBuilder(),
                developerNewIncomingChangesUpdater,
                shelvedChangesUpdater,
                enableSwitchAndShelveFeatureDialog);
        }

        bool mIsRefreshing;
        bool mShowHiddenBranches;
        bool mShouldScrollToSelection;

        long mLoadedBranchId = -1;
        object mLock = new object();

        SearchField mSearchField;
        DateFilter mDateFilter;
        BranchesListView mBranchesListView;
        BranchOperations mBranchOperations;

        LaunchTool.IProcessExecutor mProcessExecutor;
        LaunchTool.IShowDownloadPlasticExeWindow mShowDownloadPlasticExeWindow;

        readonly EmptyStateData mEmptyStateData = new EmptyStateData();
        readonly WorkspaceInfo mWkInfo;
        readonly EditorWindow mParentWindow;
        readonly ISaveAssets mSaveAssets;
        readonly IGluonUpdateReport mGluonUpdateReport;
        readonly WorkspaceWindow mWorkspaceWindow;
        readonly ViewHost mViewHost;
        readonly bool mIsGluonMode;
        readonly ProgressControlsForViews mProgressControls;
        readonly NewIncomingChangesUpdater mDeveloperNewIncomingChangesUpdater;
        readonly GluonNewIncomingChangesUpdater mGluonNewIncomingChangesUpdater;
        readonly IShelvedChangesUpdater mShelvedChangesUpdater;
        readonly WorkspaceOperationsMonitor mWorkspaceOperationsMonitor;
        readonly IShelvePendingChangesQuestionerBuilder mShelvePendingChangesQuestionerBuilder;
        readonly SwitchAndShelve.IEnableSwitchAndShelveFeatureDialog mEnableSwitchAndShelveFeatureDialog;
    }
}
