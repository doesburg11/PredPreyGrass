using System;
using System.Collections.Generic;

using UnityEditor;
using UnityEditor.IMGUI.Controls;
using UnityEngine;

using Codice.Client.Common.Threading;
using Codice.CM.Common;
using Codice.CM.Common.Mount;
using GluonGui.WorkspaceWindow.Views.WorkspaceExplorer.Explorer;
using PlasticGui;
using PlasticGui.WorkspaceWindow;
using PlasticGui.WorkspaceWindow.QueryViews;
using PlasticGui.WorkspaceWindow.QueryViews.Shelves;
using Unity.PlasticSCM.Editor.AssetUtils;
using Unity.PlasticSCM.Editor.AssetUtils.Processor;
using Unity.PlasticSCM.Editor.Tool;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.Progress;
using Unity.PlasticSCM.Editor.UI.Tree;
using Unity.PlasticSCM.Editor.Views.Changesets;
using Unity.PlasticSCM.Editor.Views.Diff;

using GluonShelveOperations = GluonGui.WorkspaceWindow.Views.Shelves.ShelveOperations;

namespace Unity.PlasticSCM.Editor.Views.Shelves
{
    internal partial class ShelvesTab :
        IRefreshableView,
        IShelveMenuOperations
    {
        internal string EmptyStateMessage { get { return mEmptyStateData.Content.text; } }
        internal ShelvesListView Table { get { return mShelvesListView; } }
        internal IShelveMenuOperations Operations { get { return this; } }
        internal IProgressControls ProgressControls { get { return mProgressControls; } }

        internal ShelvesTab(
            WorkspaceInfo wkInfo,
            RepositorySpec repSpec,
            WorkspaceWindow workspaceWindow,
            ChangesetInfo shelveToSelect,
            IViewSwitcher viewSwitcher,
            IMergeViewLauncher mergeViewLauncher,
            GluonShelveOperations.ICheckinView pendingChangesTab,
            IProgressOperationHandler progressOperationHandler,
            IUpdateProgress updateProgress,
            IHistoryViewLauncher historyViewLauncher,
            IShelvedChangesUpdater shelvedChangesUpdater,
            LaunchTool.IShowDownloadPlasticExeWindow showDownloadPlasticExeWindow,
            LaunchTool.IProcessExecutor processExecutor,
            WorkspaceOperationsMonitor workspaceOperationsMonitor,
            ISaveAssets saveAssets,
            EditorWindow parentWindow,
            bool isGluonMode)
        {
            mWkInfo = wkInfo;
            mRepSpec = repSpec;
            mRefreshView = workspaceWindow;
            mMergeViewLauncher = mergeViewLauncher;
            mPendingChangesTab = pendingChangesTab;
            mProgressOperationHandler = progressOperationHandler;
            mUpdateProgress = updateProgress;
            mShelvedChangesUpdater = shelvedChangesUpdater;
            mShowDownloadPlasticExeWindow = showDownloadPlasticExeWindow;
            mProcessExecutor = processExecutor;
            mWorkspaceOperationsMonitor = workspaceOperationsMonitor;
            mSaveAssets = saveAssets;
            mParentWindow = parentWindow;
            mIsGluonMode = isGluonMode;

            mProgressControls = new ProgressControlsForViews();

            BuildComponents(
                wkInfo,
                workspaceWindow,
                workspaceWindow,
                viewSwitcher,
                historyViewLauncher,
                parentWindow);

            mSplitterState = PlasticSplitterGUILayout.InitSplitterState(
                new float[] { 0.50f, 0.50f },
                new int[] { 100, (int)UnityConstants.DIFF_PANEL_MIN_WIDTH },
                new int[] { 100000, 100000 }
            );

            RefreshAndSelect(shelveToSelect);
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
                mShelvesListView.multiColumnHeader.state,
                UnityConstants.SHELVES_TABLE_SETTINGS_NAME);
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

            DoShelvesArea(
                mShelvesListView,
                mEmptyStateData,
                mProgressControls.IsOperationRunning(),
                mParentWindow.Repaint);

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
                mShelvesListView,
                UnityConstants.SEARCH_FIELD_WIDTH);
        }

        internal void DrawOwnerFilter()
        {
            GUI.enabled = !mProgressControls.IsOperationRunning();

            EditorGUI.BeginChangeCheck();

            mOwnerFilter = (OwnerFilter)
                EditorGUILayout.EnumPopup(
                    mOwnerFilter,
                    EditorStyles.toolbarDropDown,
                    GUILayout.Width(100));

            if (EditorGUI.EndChangeCheck())
            {
                EnumPopupSetting<OwnerFilter>.Save(
                    mOwnerFilter,
                    UnityConstants.SHELVES_OWNER_FILTER_SETTING_NAME);

                ((IRefreshableView)this).Refresh();
            }

            GUI.enabled = true;
        }

        void IRefreshableView.Refresh()
        {
            RefreshAndSelect(null);
        }

        //IQueryRefreshableView
        public void RefreshAndSelect(RepObjectInfo repObj)
        {
            List<RepObjectInfo> shelvesToSelect = repObj == null ?
                ShelvesSelection.GetSelectedRepObjectInfos(mShelvesListView) :
                new List<RepObjectInfo> { repObj };

            FillShelves(
                mWkInfo,
                QueryConstants.BuildShelvesQuery(mOwnerFilter == OwnerFilter.MyShelves),
                shelvesToSelect);
        }

        int IShelveMenuOperations.GetSelectedShelvesCount()
        {
            return ShelvesSelection.GetSelectedShelvesCount(mShelvesListView);
        }

        void IShelveMenuOperations.OpenSelectedShelveInNewWindow()
        {
            LaunchDiffOperations.DiffChangeset(
                mShowDownloadPlasticExeWindow,
                mProcessExecutor,
                ShelvesSelection.GetSelectedRepository(mShelvesListView),
                ShelvesSelection.GetSelectedShelve(mShelvesListView),
                mIsGluonMode);
        }

        void IShelveMenuOperations.ApplyShelveInWorkspace()
        {
            bool isCancelled;
            mSaveAssets.UnderWorkspaceWithConfirmation(
                mWkInfo.ClientPath, mWorkspaceOperationsMonitor,
                out isCancelled);

            if (isCancelled)
                return;

            ChangesetInfo shelveToApply = ShelvesSelection.GetSelectedShelve(mShelvesListView);

            if (mIsGluonMode)
            {
                GluonShelveOperations.ApplyPartialShelveset(
                    mWkInfo,
                    shelveToApply,
                    mRefreshView,
                    PlasticExeLauncher.BuildForResolveConflicts(
                        mWkInfo, true, mShowDownloadPlasticExeWindow),
                    this,
                    mProgressControls,
                    mPendingChangesTab,
                    mUpdateProgress,
                    mProgressOperationHandler,
                    mShelvedChangesUpdater);
                return;
            }

            ShelveOperations.ApplyShelveInWorkspace(
                mRepSpec,
                shelveToApply,
                mMergeViewLauncher,
                mProgressOperationHandler);
        }

        void IShelveMenuOperations.DeleteShelve()
        {
            ShelveOperations.DeleteShelve(
                ShelvesSelection.GetSelectedRepositories(mShelvesListView),
                ShelvesSelection.GetSelectedShelves(mShelvesListView),
                this,
                mProgressControls,
                mShelvedChangesUpdater);
        }

        void SearchField_OnDownOrUpArrowKeyPressed()
        {
            mShelvesListView.SetFocusAndEnsureSelectedItem();
        }

        void OnShelvesListViewSizeChanged()
        {
            if (!mShouldScrollToSelection)
                return;

            mShouldScrollToSelection = false;
            TableViewOperations.ScrollToSelection(mShelvesListView);
        }

        void OnSelectionChanged()
        {
            List<RepObjectInfo> selectedShelves = ShelvesSelection.
                GetSelectedRepObjectInfos(mShelvesListView);

            if (selectedShelves.Count != 1)
                return;

            mDiffPanel.UpdateInfo(
                MountPointWithPath.BuildWorkspaceRootMountPoint(
                    ShelvesSelection.GetSelectedRepository(mShelvesListView)),
                (ChangesetInfo)selectedShelves[0]);
        }

        void FillShelves(
            WorkspaceInfo wkInfo,
            string query,
            List<RepObjectInfo> shelvesToSelect)
        {
            if (mIsRefreshing)
                return;

            mIsRefreshing = true;

            int defaultRow = TableViewOperations.
                GetFirstSelectedRow(mShelvesListView);

            ((IProgressControls)mProgressControls).ShowProgress(
                PlasticLocalization.GetString(
                    PlasticLocalization.Name.LoadingShelves));

            ViewQueryResult queryResult = null;

            IThreadWaiter waiter = ThreadWaiter.GetWaiter();
            waiter.Execute(
                /*threadOperationDelegate*/ delegate
                {
                    queryResult = new ViewQueryResult(
                        PlasticGui.Plastic.API.FindQuery(wkInfo, query));
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

                        UpdateShelvesList(mShelvesListView, queryResult);

                        int shelvesCount = GetShelvesCount(queryResult);

                        if (shelvesCount == 0)
                        {
                            mDiffPanel.ClearInfo();
                            return;
                        }

                        ShelvesSelection.SelectShelves(
                            mShelvesListView, shelvesToSelect, defaultRow);
                    }
                    finally
                    {
                        ((IProgressControls)mProgressControls).HideProgress();
                        mIsRefreshing = false;
                    }
                });
        }

        static void UpdateShelvesList(
             ShelvesListView shelvesListView,
             ViewQueryResult queryResult)
        {
            shelvesListView.BuildModel(queryResult);

            shelvesListView.Refilter();

            shelvesListView.Sort();

            shelvesListView.Reload();
        }

        static int GetShelvesCount(ViewQueryResult queryResult)
        {
            if (queryResult == null)
                return 0;

            return queryResult.Count();
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

        void DoShelvesArea(
            ShelvesListView shelvesListView,
            EmptyStateData emptyStateData,
            bool isOperationRunning,
            Action repaint)
        {
            EditorGUILayout.BeginVertical();

            GUI.enabled = !isOperationRunning;

            Rect rect = GUILayoutUtility.GetRect(0, 100000, 0, 100000);

            shelvesListView.OnGUI(rect);

            emptyStateData.Update(
                GetEmptyStateMessage(shelvesListView),
                rect, Event.current.type, repaint);

            if (!emptyStateData.IsEmpty())
                DrawTreeViewEmptyState.For(emptyStateData);

            GUI.enabled = true;

            EditorGUILayout.EndVertical();
        }

        static void DoChangesArea(DiffPanel diffPanel)
        {
            EditorGUILayout.BeginVertical();

            diffPanel.OnGUI();

            EditorGUILayout.EndVertical();
        }

        static string GetEmptyStateMessage(ShelvesListView shelvesListView)
        {
            if (shelvesListView.GetRows().Count > 0)
                return string.Empty;

            return string.IsNullOrEmpty(shelvesListView.searchString) ?
                PlasticLocalization.Name.NoShelvesCreatedExplanation.GetString() :
                PlasticLocalization.Name.ShelvesEmptyState.GetString();
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

            mOwnerFilter = EnumPopupSetting<OwnerFilter>.Load(
                UnityConstants.SHELVES_OWNER_FILTER_SETTING_NAME,
                OwnerFilter.MyShelves);

            ShelvesListHeaderState headerState =
                ShelvesListHeaderState.GetDefault();

            TreeHeaderSettings.Load(
                headerState,
                UnityConstants.SHELVES_TABLE_SETTINGS_NAME,
                (int)ShelvesListColumn.Name,
                false);

            mShelvesListView = new ShelvesListView(
                headerState,
                ShelvesListHeaderState.GetColumnNames(),
                new ShelvesViewMenu(this),
                sizeChangedAction: OnShelvesListViewSizeChanged,
                selectionChangedAction: OnSelectionChanged,
                doubleClickAction: ((IShelveMenuOperations)this).OpenSelectedShelveInNewWindow);

            mShelvesListView.Reload();

            mDiffPanel = new DiffPanel(
                wkInfo, workspaceWindow, refreshView, viewSwitcher,
                historyViewLauncher, mShowDownloadPlasticExeWindow,
                parentWindow, mIsGluonMode);
        }

        internal enum OwnerFilter
        {
            MyShelves,
            AllShelves
        }

        bool mIsRefreshing;
        bool mShouldScrollToSelection;
        OwnerFilter mOwnerFilter;

        object mSplitterState;
        SearchField mSearchField;
        ShelvesListView mShelvesListView;
        DiffPanel mDiffPanel;

        readonly EmptyStateData mEmptyStateData = new EmptyStateData();
        readonly ProgressControlsForViews mProgressControls;
        readonly bool mIsGluonMode;
        readonly EditorWindow mParentWindow;
        readonly LaunchTool.IProcessExecutor mProcessExecutor;
        readonly WorkspaceOperationsMonitor mWorkspaceOperationsMonitor;
        readonly ISaveAssets mSaveAssets;
        readonly LaunchTool.IShowDownloadPlasticExeWindow mShowDownloadPlasticExeWindow;
        readonly IShelvedChangesUpdater mShelvedChangesUpdater;
        readonly IUpdateProgress mUpdateProgress;
        readonly IProgressOperationHandler mProgressOperationHandler;
        readonly GluonShelveOperations.ICheckinView mPendingChangesTab;
        readonly IMergeViewLauncher mMergeViewLauncher;
        readonly IRefreshView mRefreshView;
        readonly WorkspaceInfo mWkInfo;
        readonly RepositorySpec mRepSpec;
    }
}
