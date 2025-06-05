using System.Collections.Generic;

using UnityEditor;
using UnityEditor.IMGUI.Controls;
using UnityEngine;

using Codice.Client.Common;
using Codice.Client.Common.Threading;
using Codice.CM.Common;
using Codice.CM.Common.Mount;
using PlasticGui;
using PlasticGui.WorkspaceWindow;
using PlasticGui.WorkspaceWindow.Update;
using PlasticGui.WorkspaceWindow.QueryViews;
using PlasticGui.WorkspaceWindow.QueryViews.Changesets;
using GluonGui;
using PlasticGui.WorkspaceWindow.CodeReview;
using Unity.PlasticSCM.Editor.AssetUtils;
using Unity.PlasticSCM.Editor.AssetUtils.Processor;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.Progress;
using Unity.PlasticSCM.Editor.UI.Tree;
using Unity.PlasticSCM.Editor.Tool;
using Unity.PlasticSCM.Editor.Views.Diff;

using GluonNewIncomingChangesUpdater = PlasticGui.Gluon.WorkspaceWindow.NewIncomingChangesUpdater;
using IGluonUpdateReport = PlasticGui.Gluon.IUpdateReport;

namespace Unity.PlasticSCM.Editor.Views.Changesets
{
    internal partial class ChangesetsTab :
        IRefreshableView,
        IChangesetMenuOperations,
        ChangesetsViewMenu.IMenuOperations,
        ILaunchCodeReviewWindow
    {
        internal ChangesetsListView Table { get { return mChangesetsListView; } }
        internal IChangesetMenuOperations Operations { get { return this; } }

        internal interface IRevertToChangesetListener
        {
            void OnSuccessOperation();
        }

        internal ChangesetsTab(
            WorkspaceInfo wkInfo,
            WorkspaceWindow workspaceWindow,
            ChangesetInfo changesetToSelect,
            IViewSwitcher viewSwitcher,
            IMergeViewLauncher mergeViewLauncher,
            IHistoryViewLauncher historyViewLauncher,
            ViewHost viewHost,
            IUpdateReport updateReport,
            IGluonUpdateReport gluonUpdateReport,
            NewIncomingChangesUpdater developerNewIncomingChangesUpdater,
            GluonNewIncomingChangesUpdater gluonNewIncomingChangesUpdater,
            IShelvedChangesUpdater shelvedChangesUpdater,
            IRevertToChangesetListener revertToChangesetListener,
            LaunchTool.IShowDownloadPlasticExeWindow showDownloadPlasticExeWindow,
            LaunchTool.IProcessExecutor processExecutor,
            WorkspaceOperationsMonitor workspaceOperationsMonitor,
            ISaveAssets saveAssets,
            EditorWindow parentWindow,
            bool isGluonMode)
        {
            mWkInfo = wkInfo;
            mViewHost = viewHost;
            mWorkspaceWindow = workspaceWindow;
            mViewSwitcher = viewSwitcher;
            mRevertToChangesetListener = revertToChangesetListener;
            mShowDownloadPlasticExeWindow = showDownloadPlasticExeWindow;
            mProcessExecutor = processExecutor;
            mWorkspaceOperationsMonitor = workspaceOperationsMonitor;
            mSaveAssets = saveAssets;
            mParentWindow = parentWindow;
            mIsGluonMode = isGluonMode;
            mGluonUpdateReport = gluonUpdateReport;

            mGluonNewIncomingChangesUpdater = gluonNewIncomingChangesUpdater;
            mShelvePendingChangesQuestionerBuilder = new ShelvePendingChangesQuestionerBuilder(parentWindow);
            mShelvedChangesUpdater = shelvedChangesUpdater;
            mEnableSwitchAndShelveFeatureDialog = new EnableSwitchAndShelveFeature(
                PlasticGui.Plastic.API.GetRepositorySpec(mWkInfo),
                mParentWindow);

            BuildComponents(
                wkInfo,
                workspaceWindow,
                workspaceWindow,
                viewSwitcher,
                historyViewLauncher,
                parentWindow);

            mProgressControls = new ProgressControlsForViews();

            mSplitterState = PlasticSplitterGUILayout.InitSplitterState(
                new float[] { 0.50f, 0.50f },
                new int[] { 100, (int)UnityConstants.DIFF_PANEL_MIN_WIDTH },
                new int[] { 100000, 100000 }
            );

            mChangesetOperations = new ChangesetOperations(
                wkInfo,
                workspaceWindow,
                viewSwitcher,
                mergeViewLauncher,
                ViewType.ChangesetsView,
                mProgressControls,
                updateReport,
                new ContinueWithPendingChangesQuestionerBuilder(viewSwitcher, parentWindow),
                mShelvePendingChangesQuestionerBuilder,
                new ApplyShelveWithConflictsQuestionerBuilder(),
                developerNewIncomingChangesUpdater,
                shelvedChangesUpdater,
                null,
                null,
                mEnableSwitchAndShelveFeatureDialog);

            RefreshAndSelect(changesetToSelect);
        }

        internal void OnEnable()
        {
            mDiffPanel.OnEnable();

            mSearchField.downOrUpArrowKeyPressed +=
                SearchField_OnDownOrUpArrowKeyPressed;
        }

        internal void OnDisable()
        {
            mDiffPanel.OnDisable();

            mSearchField.downOrUpArrowKeyPressed -=
                SearchField_OnDownOrUpArrowKeyPressed;

            TreeHeaderSettings.Save(
                mChangesetsListView.multiColumnHeader.state,
                UnityConstants.CHANGESETS_TABLE_SETTINGS_NAME);
        }

        internal void Update()
        {
            mDiffPanel.Update();

            mProgressControls.UpdateProgress(mParentWindow);
        }

        internal void OnGUI()
        {
            DoActionsToolbar(mProgressControls);

            PlasticSplitterGUILayout.BeginHorizontalSplit(mSplitterState);

            DoChangesetsArea(
                mChangesetsListView,
                mProgressControls.IsOperationRunning());

            EditorGUILayout.BeginHorizontal();

            Rect border = GUILayoutUtility.GetRect(1, 0, 1, 100000);
            EditorGUI.DrawRect(border, UnityStyles.Colors.BarBorder);

            DoChangesArea(mDiffPanel);

            EditorGUILayout.EndHorizontal();

            PlasticSplitterGUILayout.EndHorizontalSplit();

        }

        internal void DrawSearchFieldForTab()
        {
            DrawSearchField.For(
                mSearchField,
                mChangesetsListView,
                UnityConstants.SEARCH_FIELD_WIDTH);
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
                    UnityConstants.CHANGESETS_DATE_FILTER_SETTING_NAME);

                ((IRefreshableView)this).Refresh();
            }

            GUI.enabled = true;
        }

        internal void SetWorkingObjectInfo(WorkingObjectInfo homeInfo)
        {
            if (mIsGluonMode)
                return;

            lock (mLock)
            {
                mLoadedChangesetId = homeInfo.GetChangesetId();
            }

            mChangesetsListView.SetLoadedChangesetId(mLoadedChangesetId);
            mChangesetsViewMenu.SetLoadedBranchId(homeInfo.BranchInfo.BranchId);
        }

        internal void SetRevertToChangesetOperationInterfacesForTesting(
            RevertToChangesetOperation.IGetStatusForWorkspace getStatusForWorkspace,
            RevertToChangesetOperation.IUndoCheckoutOperation undoCheckoutOperation,
            RevertToChangesetOperation.IRevertToChangesetMergeController revertToChangesetMergeController)
        {
            mGetStatusForWorkspace = getStatusForWorkspace;
            mUndoCheckoutOperation = undoCheckoutOperation;
            mRevertToChangesetMergeController = revertToChangesetMergeController;
        }

        internal void SetLaunchToolForTesting(
            LaunchTool.IShowDownloadPlasticExeWindow showDownloadPlasticExeWindow,
            LaunchTool.IProcessExecutor processExecutor)
        {
            mShowDownloadPlasticExeWindow = showDownloadPlasticExeWindow;
            mProcessExecutor = processExecutor;
        }

        internal void RefreshAndSelect(RepObjectInfo repObj)
        {
            List<RepObjectInfo> changesetsToSelect = repObj == null ?
                ChangesetsSelection.GetSelectedRepObjectInfos(mChangesetsListView) :
                new List<RepObjectInfo> { repObj };

            string query = GetChangesetsQuery(mDateFilter);

            FillChangesets(
                mWkInfo,
                query,
                changesetsToSelect);
        }

        void IRefreshableView.Refresh()
        {
            RefreshAndSelect(null);
        }

        int IChangesetMenuOperations.GetSelectedChangesetsCount()
        {
            return ChangesetsSelection.GetSelectedChangesetsCount(mChangesetsListView);
        }

        void IChangesetMenuOperations.DiffChangeset()
        {
            LaunchDiffOperations.DiffChangeset(
                mShowDownloadPlasticExeWindow,
                mProcessExecutor,
                ChangesetsSelection.GetSelectedRepository(mChangesetsListView),
                ChangesetsSelection.GetSelectedChangeset(mChangesetsListView),
                mIsGluonMode);
        }

        void IChangesetMenuOperations.DiffSelectedChangesets()
        {
            List<RepObjectInfo> selectedChangesets = ChangesetsSelection.
                GetSelectedRepObjectInfos(mChangesetsListView);

            if (selectedChangesets.Count < 2)
                return;

            LaunchDiffOperations.DiffSelectedChangesets(
                mShowDownloadPlasticExeWindow,
                mProcessExecutor,
                ChangesetsSelection.GetSelectedRepository(mChangesetsListView),
                (ChangesetExtendedInfo)selectedChangesets[0],
                (ChangesetExtendedInfo)selectedChangesets[1],
                mIsGluonMode);
        }

        void IChangesetMenuOperations.SwitchToChangeset()
        {
            SwitchToChangesetForMode();
        }

        void IChangesetMenuOperations.DiffWithAnotherChangeset() { }

        void IChangesetMenuOperations.CreateBranch()
        {
            CreateBranchForMode();
        }

        void IChangesetMenuOperations.LabelChangeset() { }

        void IChangesetMenuOperations.MergeChangeset()
        {
            mChangesetOperations.MergeChangeset(
                ChangesetsSelection.GetSelectedRepository(mChangesetsListView),
                ChangesetsSelection.GetSelectedChangeset(mChangesetsListView));
        }

        void IChangesetMenuOperations.CherryPickChangeset() { }

        void IChangesetMenuOperations.SubtractiveChangeset() { }

        void IChangesetMenuOperations.SubtractiveChangesetInterval() { }

        void IChangesetMenuOperations.CherryPickChangesetInterval() { }

        void IChangesetMenuOperations.MergeToChangeset() { }

        void IChangesetMenuOperations.MoveChangeset() { }

        void IChangesetMenuOperations.DeleteChangeset() { }

        void IChangesetMenuOperations.BrowseRepositoryOnChangeset() { }

        void IChangesetMenuOperations.CreateCodeReview()
        {
            RepositorySpec repSpec = ChangesetsSelection.GetSelectedRepository(mChangesetsListView);
            ChangesetInfo changesetInfo = ChangesetsSelection.GetSelectedChangeset(mChangesetsListView);

            NewCodeReviewBehavior choice = SelectNewCodeReviewBehavior.For(repSpec.Server);

            switch (choice)
            {
                case NewCodeReviewBehavior.CreateAndOpenInDesktop:
                    mChangesetOperations.CreateCodeReview(repSpec, changesetInfo, this);
                    break;
                case NewCodeReviewBehavior.RequestFromUnityCloud:
                    OpenRequestReviewPage.ForChangeset(repSpec, changesetInfo.ChangesetId);
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

        void IChangesetMenuOperations.RevertToChangeset()
        {
            if (((IChangesetMenuOperations)this).GetSelectedChangesetsCount() != 1)
                return;

            ChangesetExtendedInfo targetChangesetInfo = ((ChangesetsViewMenu.IMenuOperations)this).GetSelectedChangeset();

            RevertToChangesetOperation.RevertTo(
                mWkInfo,
                mViewSwitcher,
                mWorkspaceWindow,
                mProgressControls,
                mGetStatusForWorkspace,
                mUndoCheckoutOperation,
                mRevertToChangesetMergeController,
                GuiMessage.Get(),
                targetChangesetInfo,
                RefreshAsset.BeforeLongAssetOperation,
                RefreshAsset.AfterLongAssetOperation,
                mRevertToChangesetListener.OnSuccessOperation);
        }

        void ChangesetsViewMenu.IMenuOperations.DiffBranch()
        {
            LaunchDiffOperations.DiffBranch(
                mShowDownloadPlasticExeWindow,
                mProcessExecutor,
                ChangesetsSelection.GetSelectedRepository(mChangesetsListView),
                ChangesetsSelection.GetSelectedChangeset(mChangesetsListView),
                mIsGluonMode);
        }

        ChangesetExtendedInfo ChangesetsViewMenu.IMenuOperations.GetSelectedChangeset()
        {
            return ChangesetsSelection.GetSelectedChangeset(
                mChangesetsListView);
        }

        void SearchField_OnDownOrUpArrowKeyPressed()
        {
            mChangesetsListView.SetFocusAndEnsureSelectedItem();
        }

        void OnChangesetsListViewSizeChanged()
        {
            if (!mShouldScrollToSelection)
                return;

            mShouldScrollToSelection = false;
            TableViewOperations.ScrollToSelection(mChangesetsListView);
        }

        void OnSelectionChanged()
        {
            List<RepObjectInfo> selectedChangesets = ChangesetsSelection.
                GetSelectedRepObjectInfos(mChangesetsListView);

            if (selectedChangesets.Count != 1)
                return;

            mDiffPanel.UpdateInfo(
                MountPointWithPath.BuildWorkspaceRootMountPoint(
                    ChangesetsSelection.GetSelectedRepository(mChangesetsListView)),
                (ChangesetExtendedInfo)selectedChangesets[0]);
        }

        void FillChangesets(
            WorkspaceInfo wkInfo,
            string query,
            List<RepObjectInfo> changesetsToSelect)
        {
            if (mIsRefreshing)
                return;

            mIsRefreshing = true;

            int defaultRow = TableViewOperations.GetFirstSelectedRow(mChangesetsListView);

            ((IProgressControls)mProgressControls).ShowProgress(
                PlasticLocalization.GetString(
                    PlasticLocalization.Name.LoadingChangesets));

            ViewQueryResult queryResult = null;

            IThreadWaiter waiter = ThreadWaiter.GetWaiter();
            waiter.Execute(
                /*threadOperationDelegate*/ delegate
                {
                    queryResult = new ViewQueryResult(
                        PlasticGui.Plastic.API.FindQuery(wkInfo, query));

                    long changesetId = PlasticGui.Plastic.API.GetLoadedChangeset(wkInfo);

                    lock (mLock)
                    {
                        mLoadedChangesetId = changesetId;
                    }

                    BranchInfo currentBranch = PlasticGui.Plastic.API.GetWorkingBranch(wkInfo);

                    if (currentBranch != null)
                        mChangesetsViewMenu.SetLoadedBranchId(currentBranch.BranchId);
                },
                /*afterOperationDelegate*/ delegate
                {
                    try
                    {
                        if (waiter.Exception != null)
                        {
                            mDiffPanel.ClearInfo();

                            ExceptionsHandler.DisplayException(waiter.Exception);
                            return;
                        }

                        UpdateChangesetsList(
                            mChangesetsListView,
                            queryResult,
                            mLoadedChangesetId);

                        int changesetsCount = GetChangesetsCount(queryResult);

                        if (changesetsCount == 0)
                        {
                            mDiffPanel.ClearInfo();
                            return;
                        }

                        ChangesetsSelection.SelectChangesets(
                            mChangesetsListView, changesetsToSelect, defaultRow);
                    }
                    finally
                    {
                        ((IProgressControls)mProgressControls).HideProgress();
                        mIsRefreshing = false;
                    }
                });
        }

        static void UpdateChangesetsList(
            ChangesetsListView changesetsListView,
            ViewQueryResult queryResult,
            long loadedChangesetId)
        {
            changesetsListView.BuildModel(
                queryResult,
                loadedChangesetId);

            changesetsListView.Refilter();

            changesetsListView.Sort();

            changesetsListView.Reload();
        }

        static int GetChangesetsCount(
            ViewQueryResult queryResult)
        {
            if (queryResult == null)
                return 0;

            return queryResult.Count();
        }

        static string GetChangesetsQuery(DateFilter dateFilter)
        {
            if (dateFilter.FilterType == DateFilter.Type.AllTime)
                return QueryConstants.ChangesetsBeginningQuery;

            string whereClause = QueryConstants.GetDateWhereClause(
                dateFilter.GetTimeAgo());

            return string.Format("{0} {1}",
                QueryConstants.ChangesetsBeginningQuery,
                whereClause);
        }

        void DoActionsToolbar(ProgressControlsForViews progressControls)
        {
            EditorGUILayout.BeginHorizontal(EditorStyles.toolbar);

            if (progressControls.IsOperationRunning())
            {
                DrawProgressForViews.ForIndeterminateProgress(
                    progressControls.ProgressData);
            }

            GUILayout.FlexibleSpace();

            GUILayout.Space(2);

            EditorGUILayout.EndHorizontal();
        }

        static void DoChangesetsArea(
            ChangesetsListView changesetsListView,
            bool isOperationRunning)
        {
            EditorGUILayout.BeginVertical();

            GUI.enabled = !isOperationRunning;

            Rect rect = GUILayoutUtility.GetRect(0, 100000, 0, 100000);

            changesetsListView.OnGUI(rect);

            GUI.enabled = true;

            EditorGUILayout.EndVertical();
        }

        static void DoChangesArea(DiffPanel diffPanel)
        {
            EditorGUILayout.BeginVertical();

            diffPanel.OnGUI();

            EditorGUILayout.EndVertical();
        }

        void BuildComponents(
            WorkspaceInfo wkInfo,
            IWorkspaceWindow workspaceWindow,
            IRefreshView refreshView,
            IViewSwitcher viewSwitcher,
            IHistoryViewLauncher historyViewLauncher,
            EditorWindow parentWindow)
        {
            mSearchField = new SearchField();
            mSearchField.downOrUpArrowKeyPressed += SearchField_OnDownOrUpArrowKeyPressed;

            DateFilter.Type dateFilterType =
                EnumPopupSetting<DateFilter.Type>.Load(
                    UnityConstants.CHANGESETS_DATE_FILTER_SETTING_NAME,
                    DateFilter.Type.LastMonth);
            mDateFilter = new DateFilter(dateFilterType);

            ChangesetsListHeaderState headerState =
                ChangesetsListHeaderState.GetDefault();
            TreeHeaderSettings.Load(headerState,
                UnityConstants.CHANGESETS_TABLE_SETTINGS_NAME,
                (int)ChangesetsListColumn.CreationDate, false);

            mChangesetsViewMenu = new ChangesetsViewMenu(
                this,
                this,
                mIsGluonMode);

            mChangesetsListView = new ChangesetsListView(
                headerState,
                ChangesetsListHeaderState.GetColumnNames(),
                mChangesetsViewMenu,
                sizeChangedAction: OnChangesetsListViewSizeChanged,
                selectionChangedAction: OnSelectionChanged,
                doubleClickAction: ((IChangesetMenuOperations)this).DiffChangeset);
            mChangesetsListView.Reload();

            mDiffPanel = new DiffPanel(
                wkInfo, workspaceWindow, refreshView, viewSwitcher,
                historyViewLauncher, mShowDownloadPlasticExeWindow,
                parentWindow, mIsGluonMode);
        }

        bool mIsRefreshing;
        bool mShouldScrollToSelection;

        long mLoadedChangesetId = -1;

        object mSplitterState;
        object mLock = new object();

        DateFilter mDateFilter;
        SearchField mSearchField;
        ChangesetsListView mChangesetsListView;
        ChangesetsViewMenu mChangesetsViewMenu;
        DiffPanel mDiffPanel;

        RevertToChangesetOperation.IGetStatusForWorkspace mGetStatusForWorkspace =
            new RevertToChangesetOperation.GetStatusFromWorkspace();
        RevertToChangesetOperation.IUndoCheckoutOperation mUndoCheckoutOperation =
            new RevertToChangesetOperation.UndoCheckout();
        RevertToChangesetOperation.IRevertToChangesetMergeController mRevertToChangesetMergeController =
            new RevertToChangesetOperation.RevertToChangesetMergeController();

        ChangesetOperations mChangesetOperations;
        LaunchTool.IProcessExecutor mProcessExecutor;
        LaunchTool.IShowDownloadPlasticExeWindow mShowDownloadPlasticExeWindow;

        readonly WorkspaceOperationsMonitor mWorkspaceOperationsMonitor;
        readonly ISaveAssets mSaveAssets;
        readonly bool mIsGluonMode;
        readonly ViewHost mViewHost;
        readonly IGluonUpdateReport mGluonUpdateReport;
        readonly WorkspaceWindow mWorkspaceWindow;
        readonly IViewSwitcher mViewSwitcher;
        readonly ProgressControlsForViews mProgressControls;
        readonly IRevertToChangesetListener mRevertToChangesetListener;
        readonly EditorWindow mParentWindow;
        readonly WorkspaceInfo mWkInfo;
        readonly GluonNewIncomingChangesUpdater mGluonNewIncomingChangesUpdater;
        readonly IShelvePendingChangesQuestionerBuilder mShelvePendingChangesQuestionerBuilder;
        readonly IShelvedChangesUpdater mShelvedChangesUpdater;
        readonly SwitchAndShelve.IEnableSwitchAndShelveFeatureDialog mEnableSwitchAndShelveFeatureDialog;
    }
}
