using System;

using UnityEditor;

using Codice.Client.Common.EventTracking;
using Codice.Client.Common.Threading;
using Codice.CM.Common;
using GluonGui;
using PlasticGui;
using PlasticGui.Gluon;
using PlasticGui.WorkspaceWindow;
using PlasticGui.WorkspaceWindow.Merge;
using PlasticGui.WorkspaceWindow.QueryViews;
using Unity.PlasticSCM.Editor.AssetsOverlays.Cache;
using Unity.PlasticSCM.Editor.AssetUtils;
using Unity.PlasticSCM.Editor.AssetUtils.Processor;
using Unity.PlasticSCM.Editor.Tool;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.StatusBar;
using Unity.PlasticSCM.Editor.Views.Branches;
using Unity.PlasticSCM.Editor.Views.Changesets;
using Unity.PlasticSCM.Editor.Views.History;
using Unity.PlasticSCM.Editor.Views.IncomingChanges.Gluon;
using Unity.PlasticSCM.Editor.Views.Locks;
using Unity.PlasticSCM.Editor.Views.Merge.Developer;
using Unity.PlasticSCM.Editor.Views.Merge;
using Unity.PlasticSCM.Editor.Views.PendingChanges;
using Unity.PlasticSCM.Editor.Views.Shelves;

using GluonNewIncomingChangesUpdater = PlasticGui.Gluon.WorkspaceWindow.NewIncomingChangesUpdater;
using ObjectInfo = Codice.CM.Common.ObjectInfo;
using Codice.Client.Common;

namespace Unity.PlasticSCM.Editor
{
    [Serializable]
    internal class ViewSwitcherState
    {
        internal ViewSwitcher.SelectedTab SelectedTab;
        internal ViewSwitcher.SelectedTab PreviousSelectedTab;

        internal SerializableMergeTabState MergeTabState;
        internal SerializableBranchesTabState BranchesTabState;
        internal SerializableHistoryTabState HistoryTabState;
    }

    internal interface IShowChangesetInView
    {
        void ShowChangesetInView(ChangesetInfo changesetInfo);
    }

    internal interface IShowShelveInView
    {
        void ShowShelveInView(ChangesetInfo shelveInfo);
    }

    internal class ViewSwitcher :
        IViewSwitcher,
        IShowChangesetInView,
        IShowShelveInView,
        IMergeViewLauncher,
        IGluonViewSwitcher,
        IHistoryViewLauncher,
        MergeInProgress.IShowMergeView
    {
        internal enum SelectedTab
        {
            None = 0,
            PendingChanges = 1,
            IncomingChanges = 2,
            Changesets = 3,
            Shelves = 4,
            Branches = 5,
            Locks = 6,
            Merge = 7,
            History = 8,
        }

        internal PendingChangesTab PendingChangesTab { get; private set; }
        internal IIncomingChangesTab IncomingChangesTab { get; private set; }
        internal ChangesetsTab ChangesetsTab { get; private set; }
        internal ShelvesTab ShelvesTab { get; private set; }
        internal BranchesTab BranchesTab { get; private set; }
        internal LocksTab LocksTab { get; private set; }
        internal MergeTab MergeTab { get; private set; }
        internal HistoryTab HistoryTab { get; private set; }
        internal ViewSwitcherState State { get { return mState; } }

        internal ViewSwitcher(
            RepositorySpec repSpec,
            WorkspaceInfo wkInfo,
            ViewHost viewHost,
            bool isGluonMode,
            IAssetStatusCache assetStatusCache,
            LaunchTool.IShowDownloadPlasticExeWindow showDownloadPlasticExeWindow,
            LaunchTool.IProcessExecutor processExecutor,
            WorkspaceOperationsMonitor workspaceOperationsMonitor,
            ISaveAssets saveAssets,
            StatusBar statusBar,
            EditorWindow parentWindow)
        {
            mRepSpec = repSpec;
            mWkInfo = wkInfo;
            mViewHost = viewHost;
            mIsGluonMode = isGluonMode;
            mAssetStatusCache = assetStatusCache;
            mShowDownloadPlasticExeWindow = showDownloadPlasticExeWindow;
            mProcessExecutor = processExecutor;
            mWorkspaceOperationsMonitor = workspaceOperationsMonitor;
            mSaveAssets = saveAssets;
            mStatusBar = statusBar;
            mParentWindow = parentWindow;

            mPendingChangesTabButton = new TabButton();
            mIncomingChangesTabButton = new TabButton();
            mChangesetsTabButton = new TabButton();
            mShelvesTabButton = new TabButton();
            mBranchesTabButton = new TabButton();
            mLocksTabButton = new TabButton();
            mMergeTabButton = new TabButton();
            mHistoryTabButton = new TabButton();
        }

        internal bool IsViewSelected(SelectedTab tab)
        {
            return mState.SelectedTab == tab;
        }

        internal void SetNewIncomingChanges(
            NewIncomingChangesUpdater developerNewIncomingChangesUpdater,
            GluonNewIncomingChangesUpdater gluonNewIncomingChangesUpdater,
            StatusBar.IIncomingChangesNotification incomingChangesNotification)
        {
            mDeveloperNewIncomingChangesUpdater = developerNewIncomingChangesUpdater;
            mGluonNewIncomingChangesUpdater = gluonNewIncomingChangesUpdater;
            mIncomingChangesNotification = incomingChangesNotification;
        }

        internal void SetShelvedChanges(
            ShelvedChangesUpdater shelvedChangesUpdater,
            CheckShelvedChanges.IUpdateShelvedChangesNotification updateShelvedChanges)
        {
            mShelvedChangesUpdater = shelvedChangesUpdater;
            mUpdateShelvedChanges = updateShelvedChanges;
        }

        internal void SetWorkspaceWindow(WorkspaceWindow workspaceWindow)
        {
            mWorkspaceWindow = workspaceWindow;
        }

        internal void InitializeFromState(ViewSwitcherState state)
        {
            mState = state;

            if (mState.MergeTabState != null &&
                mState.MergeTabState.IsInitialized)
                BuildMergeViewFromState(mState.MergeTabState);

            if (mState.HistoryTabState != null &&
                mState.HistoryTabState.IsInitialized)
                BuildHistoryViewFromState(mState.HistoryTabState);

            if (mState.BranchesTabState != null &&
                mState.BranchesTabState.IsInitialized)
                BuildBranchesViewFromState(mState.BranchesTabState);

            ShowInitialView(mState.SelectedTab);
        }

        internal void AutoRefreshPendingChangesView()
        {
            AutoRefresh.PendingChangesView(PendingChangesTab);
        }

        internal void AutoRefreshIncomingChangesView()
        {
            AutoRefresh.IncomingChangesView(IncomingChangesTab);
        }

        internal void AutoRefreshMergeView()
        {
            if (mIsGluonMode)
                return;

            AutoRefresh.IncomingChangesView(MergeTab);
        }

        internal void RefreshView(ViewType viewType)
        {
            IRefreshableView view = GetRefreshableView(viewType);

            if (view == null)
            {
                if (viewType.Equals(ViewType.PendingChangesView))
                    PlasticPlugin.AssetStatusCache.Clear();

                if (viewType.Equals(ViewType.LocksView))
                    PlasticPlugin.AssetStatusCache.ClearLocks();

                return;
            }

            view.Refresh();
        }

        internal void RefreshSelectedView()
        {
            IRefreshableView view = GetRefreshableViewBasedOnSelectedTab(mState.SelectedTab);

            if (view == null)
                return;

            view.Refresh();
        }

        internal void RefreshWorkingObjectInfoForSelectedView(
            ViewType viewType,
            WorkingObjectInfo homeInfo)
        {
            switch (viewType)
            {
                case ViewType.BranchesView:
                    if (BranchesTab != null)
                        BranchesTab.SetWorkingObjectInfo(homeInfo);
                    break;
                case ViewType.ChangesetsView:
                    if (ChangesetsTab != null)
                        ChangesetsTab.SetWorkingObjectInfo(homeInfo);
                    break;
            }
        }

        internal void OnEnable()
        {
            if (PendingChangesTab != null)
                PendingChangesTab.OnEnable();

            if (IncomingChangesTab != null)
                IncomingChangesTab.OnEnable();

            if (ChangesetsTab != null)
                ChangesetsTab.OnEnable();

            if (ShelvesTab != null)
                ShelvesTab.OnEnable();

            if (BranchesTab != null)
                BranchesTab.OnEnable();

            if (LocksTab != null)
                LocksTab.OnEnable();

            if (MergeTab != null)
                MergeTab.OnEnable();

            if (HistoryTab != null)
                HistoryTab.OnEnable();
        }

        internal void OnDisable()
        {
            if (PendingChangesTab != null)
            {
                PendingChangesTab.OnDisable();
            }

            if (IncomingChangesTab != null)
            {
                IncomingChangesTab.OnDisable();
            }

            if (ChangesetsTab != null)
            {
                ChangesetsTab.OnDisable();
            }

            if (ShelvesTab != null)
            {
                ShelvesTab.OnDisable();
            }

            if (BranchesTab != null)
            {
                mState.BranchesTabState = BranchesTab.GetSerializableState();
                BranchesTab.OnDisable();
            }

            if (LocksTab != null)
            {
                LocksTab.OnDisable();
            }

            if (MergeTab != null)
            {
                mState.MergeTabState = MergeTab.GetSerializableState();
                MergeTab.OnDisable();
            }

            if (HistoryTab != null)
            {
                mState.HistoryTabState = HistoryTab.GetSerializableState();
                HistoryTab.OnDisable();
            }
        }

        internal void Update()
        {
            if (IsViewSelected(SelectedTab.PendingChanges))
            {
                PendingChangesTab.Update();
                return;
            }

            if (IsViewSelected(SelectedTab.IncomingChanges))
            {
                IncomingChangesTab.Update();
                return;
            }

            if (IsViewSelected(SelectedTab.Changesets))
            {
                ChangesetsTab.Update();
                return;
            }

            if (IsViewSelected(SelectedTab.Shelves))
            {
                ShelvesTab.Update();
                return;
            }

            if (IsViewSelected(SelectedTab.Branches))
            {
                BranchesTab.Update();
                return;
            }

            if (IsViewSelected(SelectedTab.Locks))
            {
                LocksTab.Update();
                return;
            }

            if (IsViewSelected(SelectedTab.Merge))
            {
                MergeTab.Update();
                return;
            }

            if (IsViewSelected(SelectedTab.History))
            {
                HistoryTab.Update();
                return;
            }
        }

        internal void TabButtonsGUI()
        {
            InitializeTabButtonWidth();

            PendingChangesTabButtonGUI();

            IncomingChangesTabButtonGUI();

            ChangesetsTabButtonGUI();

            ShelvesTabButtonGUI();

            BranchesTabButtonGUI();

            LocksTabButtonGUI();

            MergeTabButtonGUI();

            HistoryTabButtonGUI();
        }

        internal void TabViewGUI(ResolvedUser currentUser)
        {
            if (IsViewSelected(SelectedTab.PendingChanges))
            {
                PendingChangesTab.OnGUI(
                    currentUser,
                    mParentWindow.Repaint);
                return;
            }

            if (IsViewSelected(SelectedTab.IncomingChanges))
            {
                IncomingChangesTab.OnGUI();
                return;
            }

            if (IsViewSelected(SelectedTab.Changesets))
            {
                ChangesetsTab.OnGUI();
                return;
            }

            if (IsViewSelected(SelectedTab.Shelves))
            {
                ShelvesTab.OnGUI();
                return;
            }

            if (IsViewSelected(SelectedTab.Branches))
            {
                BranchesTab.OnGUI();
                return;
            }

            if (IsViewSelected(SelectedTab.Locks))
            {
                LocksTab.OnGUI();
                return;
            }

            if (IsViewSelected(SelectedTab.Merge))
            {
                MergeTab.OnGUI();
                return;
            }

            if (IsViewSelected(SelectedTab.History))
            {
                HistoryTab.OnGUI();
                return;
            }
        }

        internal void ShowPendingChangesView()
        {
            OpenPendingChangesTab();

            bool wasPendingChangesSelected =
                IsViewSelected(SelectedTab.PendingChanges);

            if (!wasPendingChangesSelected)
            {
                PendingChangesTab.AutoRefresh();
            }

            SetSelectedView(SelectedTab.PendingChanges);
        }

        internal void ShowChangesetsView(ChangesetInfo changesetToSelect = null)
        {
            bool shouldRefreshView = ShouldRefreshView(
                ChangesetsTab != null,
                changesetToSelect != null,
                IsViewSelected(SelectedTab.Changesets));

            if (ChangesetsTab == null)
            {
                OpenPendingChangesTab();

                ChangesetsTab = new ChangesetsTab(
                    mWkInfo,
                    mWorkspaceWindow,
                    changesetToSelect,
                    this,
                    this,
                    this,
                    mViewHost,
                    mWorkspaceWindow,
                    mWorkspaceWindow,
                    mDeveloperNewIncomingChangesUpdater,
                    mGluonNewIncomingChangesUpdater,
                    mShelvedChangesUpdater,
                    PendingChangesTab,
                    mShowDownloadPlasticExeWindow,
                    mProcessExecutor,
                    mWorkspaceOperationsMonitor,
                    mSaveAssets,
                    mParentWindow,
                    mIsGluonMode);

                mViewHost.AddRefreshableView(
                    ViewType.ChangesetsView,
                    ChangesetsTab);
            }

            if (shouldRefreshView)
                ChangesetsTab.RefreshAndSelect(changesetToSelect);

            SetSelectedView(SelectedTab.Changesets);
        }

        internal void ShowShelvesViewIfNeeded()
        {
            if (!BoolSetting.Load(UnityConstants.SHOW_SHELVES_VIEW_KEY_NAME, false))
                return;

            OpenShelvesTab();
        }

        internal void ShowShelvesView(ChangesetInfo shelveToSelect = null)
        {
            bool shouldRefreshView = ShouldRefreshView(
                ShelvesTab != null,
                shelveToSelect != null,
                IsViewSelected(SelectedTab.Shelves));

            OpenShelvesTab(shelveToSelect);

            if (shouldRefreshView)
                ShelvesTab.RefreshAndSelect(shelveToSelect);

            SetSelectedView(SelectedTab.Shelves);
        }

        internal void ShowLocksViewIfNeeded()
        {
            if (!BoolSetting.Load(UnityConstants.SHOW_LOCKS_VIEW_KEY_NAME, false))
                return;

            OpenLocksTab();
        }

        internal void ShowLocksView()
        {
            OpenLocksTab();

            bool wasLocksViewSelected =
                IsViewSelected(SelectedTab.Locks);

            if (!wasLocksViewSelected)
                ((IRefreshableView)LocksTab).Refresh();

            SetSelectedView(SelectedTab.Locks);
        }

        internal void ShowBranchesViewIfNeeded()
        {
            if (!BoolSetting.Load(UnityConstants.SHOW_BRANCHES_VIEW_KEY_NAME, true))
                return;

            string query = QueryConstants.BranchesBeginningQuery;

            ViewQueryResult queryResult = null;

            IThreadWaiter waiter = ThreadWaiter.GetWaiter();
            waiter.Execute(
                /*threadOperationDelegate*/ delegate
                {
                    queryResult = new ViewQueryResult(
                        PlasticGui.Plastic.API.FindQuery(mWkInfo, query));
                },
                /*afterOperationDelegate*/ delegate
                {
                    if (waiter.Exception != null)
                    {
                        ExceptionsHandler.DisplayException(waiter.Exception);
                        return;
                    }

                    if (queryResult == null)
                        return;

                    if (queryResult.Count()>0)
                        OpenBranchesTab();
                });
        }

        internal void ShowBranchesView()
        {
            OpenBranchesTab();

            bool wasBranchesSelected =
                IsViewSelected(SelectedTab.Branches);

            if (!wasBranchesSelected)
                ((IRefreshableView)BranchesTab).Refresh();

            SetSelectedView(SelectedTab.Branches);
        }

        internal void ShowHistoryView(
            RepositorySpec repSpec,
            long itemId,
            string path,
            bool isDirectory)
        {
            if (HistoryTab == null)
            {
                HistoryTab = BuildHistoryTab(
                    repSpec, itemId, path, isDirectory);

                mViewHost.AddRefreshableView(
                    ViewType.HistoryView, HistoryTab);
            }
            else
            {
                HistoryTab.RefreshForItem(repSpec, itemId, path, isDirectory);
            }

            SetSelectedView(SelectedTab.History);
        }

        internal void ShowBranchesViewForTesting(BranchesTab branchesTab)
        {
            BranchesTab = branchesTab;

            ShowBranchesView();
        }

        internal void ShowMergeViewForTesting(MergeTab mergeTab)
        {
            MergeTab = mergeTab;

            ShowMergeView();
        }

        internal void ShowShelvesViewForTesting(ShelvesTab shelvesTab)
        {
            ShelvesTab = shelvesTab;

            ShowShelvesView();
        }

        void IViewSwitcher.ShowView(ViewType viewType)
        {
        }

        void IViewSwitcher.ShowPendingChanges()
        {
            ShowPendingChangesView();
            mParentWindow.Repaint();
        }

        void IViewSwitcher.ShowShelvesView()
        {
            ShowShelvesView();
        }

        void IViewSwitcher.ShowSyncView(string syncViewToSelect)
        {
            throw new NotImplementedException();
        }

        void IViewSwitcher.ShowBranchExplorerView()
        {
            //TODO: Codice
            //launch plastic with branch explorer view option
        }

        void IViewSwitcher.DisableMergeView()
        {
            DisableMergeTab();
        }

        IMergeView IViewSwitcher.GetMergeView()
        {
            return MergeTab;
        }

        bool IViewSwitcher.IsIncomingChangesView()
        {
            return IsViewSelected(SelectedTab.IncomingChanges);
        }

        void IViewSwitcher.CloseMergeView()
        {
            CloseMergeTab();
        }

        void IShowChangesetInView.ShowChangesetInView(ChangesetInfo changesetInfo)
        {
            ShowChangesetsView(changesetInfo);
        }

        void IShowShelveInView.ShowShelveInView(ChangesetInfo shelveInfo)
        {
            ShowShelvesView(shelveInfo);
        }

        IMergeView IMergeViewLauncher.MergeFrom(
            RepositorySpec repSpec,
            ObjectInfo objectInfo,
            EnumMergeType mergeType,
            bool showDiscardChangesButton)
        {
            return ((IMergeViewLauncher)this).MergeFromInterval(
                repSpec, objectInfo, null, mergeType, showDiscardChangesButton);
        }

        IMergeView IMergeViewLauncher.MergeFrom(
            RepositorySpec repSpec,
            ObjectInfo objectInfo,
            EnumMergeType mergeType,
            ShowIncomingChangesFrom from,
            bool showDiscardChangesButton)
        {
            return MergeFromInterval(repSpec, objectInfo, null, mergeType, from, showDiscardChangesButton);
        }

        IMergeView IMergeViewLauncher.MergeFromInterval(
            RepositorySpec repSpec,
            ObjectInfo objectInfo,
            ObjectInfo ancestorChangesetInfo,
            EnumMergeType mergeType,
            bool showDiscardChangesButton)
        {
            return MergeFromInterval(
                repSpec, objectInfo, null, mergeType, ShowIncomingChangesFrom.NotificationBar, showDiscardChangesButton);
        }

        IMergeView IMergeViewLauncher.FromCalculatedMerge(
            RepositorySpec repSpec,
            ObjectInfo objectInfo,
            EnumMergeType mergeType,
            CalculatedMergeResult calculatedMergeResult,
            bool showDiscardChangesButton)
        {
            return ShowMergeViewFromCalculatedMerge(
                repSpec, objectInfo, mergeType, calculatedMergeResult, showDiscardChangesButton);
        }

        void IGluonViewSwitcher.ShowIncomingChangesView()
        {
            ShowIncomingChangesView();
            mParentWindow.Repaint();
        }

        void IHistoryViewLauncher.ShowHistoryView(
            RepositorySpec repSpec,
            long itemId,
            string path,
            bool isDirectory)
        {
            ShowHistoryView(
                repSpec,
                itemId,
                path,
                isDirectory);

            mParentWindow.Repaint();
        }

        void MergeInProgress.IShowMergeView.MergeLinkNotFound()
        {
            // Nothing to do on the plugin when there is no pending merge link
        }

        void MergeInProgress.IShowMergeView.ForPendingMergeLink(
            RepositorySpec repSpec,
            MergeType pendingLinkMergeType,
            ChangesetInfo srcChangeset,
            ChangesetInfo baseChangeset)
        {
            EnumMergeType mergeType = MergeTypeConverter.TranslateMergeType(pendingLinkMergeType);

            MergeTab = BuildMergeTab(
                repSpec,
                srcChangeset,
                baseChangeset,
                mergeType,
                ShowIncomingChangesFrom.None,
                MergeTypeClassifier.IsIncomingMerge(mergeType),
                false,
                false);

            mViewHost.AddRefreshableView(ViewType.MergeView, MergeTab);

            ShowMergeView();
        }

        void ShowInitialView(SelectedTab viewToShow)
        {
            mState.SelectedTab = SelectedTab.None;

            ShowView(viewToShow);

            if (mState.SelectedTab != SelectedTab.None)
                return;

            ShowPendingChangesView();
        }

        void BuildHistoryViewFromState(SerializableHistoryTabState state)
        {
            HistoryTab = BuildHistoryTab(
                state.RepSpec,
                state.ItemId,
                state.Path,
                state.IsDirectory);

            mViewHost.AddRefreshableView(ViewType.HistoryView, HistoryTab);
        }

        void BuildMergeViewFromState(SerializableMergeTabState state)
        {
            MergeTab = BuildMergeTab(
                state.RepSpec,
                state.GetObjectInfo(),
                state.GetAncestorObjectInfo(),
                state.MergeType,
                state.From,
                state.IsIncomingMerge,
                state.IsMergeFinished,
                false);

            mViewHost.AddRefreshableView(ViewType.MergeView, MergeTab);
        }

        void BuildBranchesViewFromState(SerializableBranchesTabState state)
        {
            BranchesTab = BuildBranchesTab(
                state.ShowHiddenBranches);

            mViewHost.AddRefreshableView(ViewType.BranchesView, BranchesTab);
        }

        void OpenPendingChangesTab()
        {
            if (PendingChangesTab != null)
                return;

            PendingChangesTab = new PendingChangesTab(
                mWkInfo,
                mRepSpec,
                mViewHost,
                mIsGluonMode,
                mWorkspaceWindow,
                this,
                this,
                this,
                this,
                this,
                mShowDownloadPlasticExeWindow,
                mWorkspaceOperationsMonitor,
                mSaveAssets,
                mDeveloperNewIncomingChangesUpdater,
                mGluonNewIncomingChangesUpdater,
                mAssetStatusCache,
                mStatusBar,
                mParentWindow);

            mViewHost.AddRefreshableView(
                ViewType.CheckinView,
                PendingChangesTab);
        }

        IMergeView MergeFromInterval(
            RepositorySpec repSpec,
            ObjectInfo objectInfo,
            ObjectInfo ancestorChangesetInfo,
            EnumMergeType mergeType,
            ShowIncomingChangesFrom from,
            bool showDiscardChangesButton)
        {
            if (MergeTypeClassifier.IsIncomingMerge(mergeType))
            {
                ShowIncomingChangesView();
                mParentWindow.Repaint();
                return IncomingChangesTab as IMergeView;
            }

            ShowMergeViewFromInterval(
                repSpec, objectInfo, ancestorChangesetInfo, mergeType, from, showDiscardChangesButton);
            mParentWindow.Repaint();
            return MergeTab;
        }

        void OpenShelvesTab(ChangesetInfo shelveToSelect = null)
        {
            if (ShelvesTab == null)
            {
                OpenPendingChangesTab();

                ShelvesTab = new ShelvesTab(
                     mWkInfo,
                     mRepSpec,
                     mWorkspaceWindow,
                     shelveToSelect,
                     this,
                     this,
                     PendingChangesTab,
                     mIsGluonMode ?
                         mWorkspaceWindow.GluonProgressOperationHandler :
                         mWorkspaceWindow.DeveloperProgressOperationHandler,
                     mWorkspaceWindow.GluonProgressOperationHandler,
                     this,
                     mShelvedChangesUpdater,
                     mShowDownloadPlasticExeWindow,
                     mProcessExecutor,
                     mWorkspaceOperationsMonitor,
                     mSaveAssets,
                     mParentWindow,
                     mIsGluonMode);

                mViewHost.AddRefreshableView(ViewType.ShelvesView, ShelvesTab);

                TrackFeatureUseEvent.For(
                    mRepSpec, TrackFeatureUseEvent.Features.OpenShelvesView);
            }

            BoolSetting.Save(true, UnityConstants.SHOW_SHELVES_VIEW_KEY_NAME);
        }

        void OpenBranchesTab()
        {
            if (BranchesTab == null)
            {
                BranchesTab = BuildBranchesTab(false);

                mViewHost.AddRefreshableView(ViewType.BranchesView, BranchesTab);

                TrackFeatureUseEvent.For(
                   mRepSpec, TrackFeatureUseEvent.Features.OpenBranchesView);
            }

            BoolSetting.Save(true, UnityConstants.SHOW_BRANCHES_VIEW_KEY_NAME);
        }

        void OpenLocksTab()
        {
            if (LocksTab == null)
            {
                LocksTab = new LocksTab(
                    mRepSpec,
                    mWorkspaceWindow,
                    mAssetStatusCache,
                    mParentWindow);

                mViewHost.AddRefreshableView(ViewType.LocksView, LocksTab);

                TrackFeatureUseEvent.For(mRepSpec,
                    TrackFeatureUseEvent.Features.OpenLocksView);
            }

            BoolSetting.Save(true, UnityConstants.SHOW_LOCKS_VIEW_KEY_NAME);
        }

        void ShowHistoryView()
        {
            if (HistoryTab == null)
                return;

            ((IRefreshableView)HistoryTab).Refresh();

            SetSelectedView(SelectedTab.History);
        }

        void ShowIncomingChangesView()
        {
            if (IncomingChangesTab == null)
            {
                IncomingChangesTab = BuildIncomingChangesTab(mIsGluonMode);

                mViewHost.AddRefreshableView(
                    ViewType.IncomingChangesView,
                    (IRefreshableView)IncomingChangesTab);
            }

            bool wasIncomingChangesSelected =
                IsViewSelected(SelectedTab.IncomingChanges);

            if (!wasIncomingChangesSelected)
                IncomingChangesTab.AutoRefresh();

            SetSelectedView(SelectedTab.IncomingChanges);
        }

        void ShowMergeViewFromInterval(
            RepositorySpec repSpec,
            ObjectInfo objectInfo,
            ObjectInfo ancestorChangesetInfo,
            EnumMergeType mergeType,
            ShowIncomingChangesFrom from,
            bool showDiscardChangesButton)
        {
            if (MergeTab != null && MergeTab.IsProcessingMerge)
            {
                ShowMergeView();
                return;
            }

            if (MergeTab != null)
            {
                mViewHost.RemoveRefreshableView(ViewType.MergeView, MergeTab);
                MergeTab.OnDisable();
            }

            MergeTab = BuildMergeTab(
                repSpec,
                objectInfo,
                ancestorChangesetInfo,
                mergeType,
                from,
                false,
                false,
                showDiscardChangesButton);

            mViewHost.AddRefreshableView(ViewType.MergeView, MergeTab);

            ShowMergeView();
        }

        IMergeView ShowMergeViewFromCalculatedMerge(
            RepositorySpec repSpec,
            ObjectInfo objectInfo,
            EnumMergeType mergeType,
            CalculatedMergeResult calculatedMergeResult,
            bool showDiscardChangesButton)
        {
            if (MergeTab != null && MergeTab.IsProcessingMerge)
            {
                ShowMergeView();
                mParentWindow.Repaint();
                return MergeTab;
            }

            if (MergeTab != null)
            {
                mViewHost.RemoveRefreshableView(ViewType.MergeView, MergeTab);
                MergeTab.OnDisable();
            }

            MergeTab = BuildMergeTabFromCalculatedMerge(
                repSpec, objectInfo, mergeType, calculatedMergeResult, showDiscardChangesButton);

            mViewHost.AddRefreshableView(ViewType.MergeView, MergeTab);

            ShowMergeView();
            mParentWindow.Repaint();
            return MergeTab;
        }

        void ShowMergeView()
        {
            if (MergeTab == null)
                return;

            bool wasMergeTabSelected =
                IsViewSelected(SelectedTab.Merge);

            if (!wasMergeTabSelected)
                MergeTab.AutoRefresh();

            SetSelectedView(SelectedTab.Merge);
        }

        void DisableMergeTab()
        {
            if (MergeTab == null)
                return;

            mViewHost.RemoveRefreshableView(
                ViewType.MergeView, MergeTab);

            MergeTab.OnDisable();
            MergeTab = null;

            mState.MergeTabState = null;
        }

        void CloseShelvesTab()
        {
            BoolSetting.Save(false, UnityConstants.SHOW_SHELVES_VIEW_KEY_NAME);

            TrackFeatureUseEvent.For(
                mRepSpec, TrackFeatureUseEvent.Features.CloseShelvesView);

            mViewHost.RemoveRefreshableView(
                ViewType.ShelvesView, ShelvesTab);

            ShelvesTab.OnDisable();
            ShelvesTab = null;

            ShowPreviousViewFrom(SelectedTab.Shelves);

            mParentWindow.Repaint();
        }

        void CloseBranchesTab()
        {
            BoolSetting.Save(false, UnityConstants.SHOW_BRANCHES_VIEW_KEY_NAME);

            mViewHost.RemoveRefreshableView(
                ViewType.BranchesView, BranchesTab);

            BranchesTab.OnDisable();
            BranchesTab = null;

            mState.BranchesTabState = null;

            ShowPreviousViewFrom(SelectedTab.Branches);

            mParentWindow.Repaint();
        }

        void CloseLocksTab()
        {
            BoolSetting.Save(false, UnityConstants.SHOW_LOCKS_VIEW_KEY_NAME);

            TrackFeatureUseEvent.For(
                mRepSpec, TrackFeatureUseEvent.Features.CloseLocksView);

            mViewHost.RemoveRefreshableView(ViewType.LocksView, LocksTab);

            LocksTab.OnDisable();
            LocksTab = null;

            ShowPreviousViewFrom(SelectedTab.Locks);

            mParentWindow.Repaint();
        }

        void CloseMergeTab()
        {
            DisableMergeTab();

            ShowPreviousViewFrom(SelectedTab.Merge);

            mParentWindow.Repaint();
        }

        void CloseHistoryTab()
        {
            mViewHost.RemoveRefreshableView(
                ViewType.HistoryView, HistoryTab);

            HistoryTab.OnDisable();
            HistoryTab = null;

            mState.HistoryTabState = null;

            ShowPreviousViewFrom(SelectedTab.History);

            mParentWindow.Repaint();
        }

        void InitializeTabButtonWidth()
        {
            if (mTabButtonWidth != -1)
                return;

            mTabButtonWidth = MeasureMaxWidth.ForTexts(
                UnityStyles.PlasticWindow.TabButton,
                PlasticLocalization.GetString(PlasticLocalization.Name.PendingChangesViewTitle),
                PlasticLocalization.GetString(PlasticLocalization.Name.IncomingChangesViewTitle),
                PlasticLocalization.GetString(PlasticLocalization.Name.ChangesetsViewTitle),
                PlasticLocalization.GetString(PlasticLocalization.Name.BranchesViewTitle),
                PlasticLocalization.GetString(PlasticLocalization.Name.ShelvesViewTitle),
                PlasticLocalization.GetString(PlasticLocalization.Name.LocksViewTitle),
                PlasticLocalization.GetString(PlasticLocalization.Name.History));
        }

        IIncomingChangesTab BuildIncomingChangesTab(bool isGluonMode)
        {
            if (isGluonMode)
            {
                return new IncomingChangesTab(
                    mWkInfo,
                    mViewHost,
                    mWorkspaceWindow,
                    mShowDownloadPlasticExeWindow,
                    mGluonNewIncomingChangesUpdater,
                    (Gluon.IncomingChangesNotification)mIncomingChangesNotification,
                    mStatusBar,
                    mParentWindow);
            }

            PlasticNotifier plasticNotifier = new PlasticNotifier();

            MergeViewLogic.IMergeController mergeController = new MergeController(
                mWkInfo,
                mRepSpec,
                null,
                null,
                EnumMergeType.IncomingMerge,
                true,
                plasticNotifier);

            return MergeTab.Build(
                mWkInfo,
                mRepSpec,
                mWorkspaceWindow,
                this,
                mShowDownloadPlasticExeWindow,
                this,
                mDeveloperNewIncomingChangesUpdater,
                mStatusBar,
                mParentWindow,
                null,
                null,
                EnumMergeType.IncomingMerge,
                ShowIncomingChangesFrom.NotificationBar,
                plasticNotifier,
                mergeController,
                new MergeViewLogic.GetWorkingBranch(),
                mUpdateShelvedChanges,
                mShelvedChangesUpdater,
                mWorkspaceWindow,
                true,
                false,
                false);
        }

        HistoryTab BuildHistoryTab(
            RepositorySpec repSpec,
            long itemId,
            string path,
            bool isDirectory)
        {
            HistoryTab result = new HistoryTab(
                mWkInfo,
                mWorkspaceWindow,
                mShowDownloadPlasticExeWindow,
                mProcessExecutor,
                mDeveloperNewIncomingChangesUpdater,
                mViewHost,
                mParentWindow,
                mIsGluonMode);

            result.RefreshForItem(repSpec, itemId, path, isDirectory);

            return result;
        }

        BranchesTab BuildBranchesTab(bool showHiddenBranches)
        {
            BranchesTab result = new BranchesTab(
                mWkInfo,
                mWorkspaceWindow,
                this,
                this,
                mViewHost,
                mWorkspaceWindow,
                mWorkspaceWindow,
                mDeveloperNewIncomingChangesUpdater,
                mGluonNewIncomingChangesUpdater,
                mShelvedChangesUpdater,
                mShowDownloadPlasticExeWindow,
                mProcessExecutor,
                mWorkspaceOperationsMonitor,
                mSaveAssets,
                mParentWindow,
                mIsGluonMode,
                showHiddenBranches);

            return result;
        }

        MergeTab BuildMergeTabFromCalculatedMerge(
            RepositorySpec repSpec,
            ObjectInfo objectInfo,
            EnumMergeType mergeType,
            CalculatedMergeResult calculatedMergeResult,
            bool showDiscardChangesButton)
        {
            return BuildMergeTab(
                repSpec,
                objectInfo,
                null,
                mergeType,
                ShowIncomingChangesFrom.None,
                false,
                false,
                showDiscardChangesButton,
                calculatedMergeResult);
        }

        MergeTab BuildMergeTab(
            RepositorySpec repSpec,
            ObjectInfo objectInfo,
            ObjectInfo ancestorObjectInfo,
            EnumMergeType mergeType,
            ShowIncomingChangesFrom from,
            bool isIncomingMerge,
            bool isMergeFinished,
            bool showDiscardChangesButton,
            CalculatedMergeResult calculatedMergeResult = null)
        {
            PlasticNotifier plasticNotifier = new PlasticNotifier();

            MergeViewLogic.IMergeController mergeController = new MergeController(
                mWkInfo,
                repSpec,
                objectInfo,
                ancestorObjectInfo,
                mergeType,
                false,
                plasticNotifier);

            if (calculatedMergeResult != null)
            {
                return MergeTab.BuildFromCalculatedMerge(
                    mWkInfo,
                    repSpec,
                    mWorkspaceWindow,
                    this,
                    mShowDownloadPlasticExeWindow,
                    this,
                    mDeveloperNewIncomingChangesUpdater,
                    mStatusBar,
                    mParentWindow,
                    objectInfo,
                    ancestorObjectInfo,
                    mergeType,
                    from,
                    plasticNotifier,
                    mergeController,
                    new MergeViewLogic.GetWorkingBranch(),
                    mUpdateShelvedChanges,
                    mShelvedChangesUpdater,
                    mWorkspaceWindow,
                    isIncomingMerge,
                    isMergeFinished,
                    calculatedMergeResult,
                    showDiscardChangesButton);
            }

            return MergeTab.Build(
                mWkInfo,
                repSpec,
                mWorkspaceWindow,
                this,
                mShowDownloadPlasticExeWindow,
                this,
                mDeveloperNewIncomingChangesUpdater,
                mStatusBar,
                mParentWindow,
                objectInfo,
                ancestorObjectInfo,
                mergeType,
                from,
                plasticNotifier,
                mergeController,
                new MergeViewLogic.GetWorkingBranch(),
                mUpdateShelvedChanges,
                mShelvedChangesUpdater,
                mWorkspaceWindow,
                isIncomingMerge,
                isMergeFinished,
                showDiscardChangesButton);
        }

        void ShowView(SelectedTab viewToShow)
        {
            switch (viewToShow)
            {
                case SelectedTab.PendingChanges:
                    ShowPendingChangesView();
                    break;

                case SelectedTab.IncomingChanges:
                    ShowIncomingChangesView();
                    break;

                case SelectedTab.Changesets:
                    ShowChangesetsView();
                    break;

                case SelectedTab.Branches:
                    ShowBranchesView();
                    break;

                case SelectedTab.Shelves:
                    ShowShelvesView();
                    break;

                case SelectedTab.Locks:
                    ShowLocksView();
                    break;

                case SelectedTab.Merge:
                    ShowMergeView();
                    break;

                case SelectedTab.History:
                    ShowHistoryView();
                    break;
            }
        }

        void ShowPreviousViewFrom(SelectedTab tabToClose)
        {
            if (!IsViewSelected(tabToClose))
                return;

            if (GetRefreshableViewBasedOnSelectedTab(mState.PreviousSelectedTab) == null)
                mState.PreviousSelectedTab = SelectedTab.PendingChanges;

            ShowView(mState.PreviousSelectedTab);
        }

        IRefreshableView GetRefreshableViewBasedOnSelectedTab(SelectedTab selectedTab)
        {
            switch (selectedTab)
            {
                case SelectedTab.PendingChanges:
                    return PendingChangesTab;

                case SelectedTab.IncomingChanges:
                    return (IRefreshableView)IncomingChangesTab;

                case SelectedTab.Changesets:
                    return ChangesetsTab;

                case SelectedTab.Shelves:
                    return ShelvesTab;

                case SelectedTab.Branches:
                    return BranchesTab;

                case SelectedTab.Locks:
                    return LocksTab;

                case SelectedTab.Merge:
                    return MergeTab;

                case SelectedTab.History:
                    return HistoryTab;

                default:
                    return null;
            }
        }

        IRefreshableView GetRefreshableView(ViewType viewType)
        {
            switch (viewType)
            {
                case ViewType.PendingChangesView:
                    return PendingChangesTab;

                case ViewType.IncomingChangesView:
                    return (IRefreshableView)IncomingChangesTab;

                case ViewType.ChangesetsView:
                    return ChangesetsTab;

                case ViewType.ShelvesView:
                    return ShelvesTab;

                case ViewType.BranchesView:
                    return BranchesTab;

                case ViewType.LocksView:
                    return LocksTab;

                case ViewType.MergeView:
                    return MergeTab;

                case ViewType.HistoryView:
                    return HistoryTab;

                default:
                    return null;
            }
        }

        void SetSelectedView(SelectedTab tab)
        {
            if (mState.SelectedTab != tab && mState.SelectedTab != SelectedTab.None)
                mState.PreviousSelectedTab = mState.SelectedTab;

            mState.SelectedTab = tab;

            if (IncomingChangesTab == null)
                return;

            IncomingChangesTab.IsVisible =
                tab == SelectedTab.IncomingChanges;
        }

        void PendingChangesTabButtonGUI()
        {
            bool wasPendingChangesSelected =
                IsViewSelected(SelectedTab.PendingChanges);

            bool isPendingChangesSelected = mPendingChangesTabButton.
                DrawTabButton(
                    PlasticLocalization.GetString(PlasticLocalization.Name.PendingChangesViewTitle),
                    wasPendingChangesSelected,
                    mTabButtonWidth);

            if (isPendingChangesSelected)
                ShowPendingChangesView();
        }

        void IncomingChangesTabButtonGUI()
        {
            bool wasIncomingChangesSelected =
                IsViewSelected(SelectedTab.IncomingChanges);

            bool isIncomingChangesSelected = mIncomingChangesTabButton.
                DrawTabButton(
                    PlasticLocalization.GetString(PlasticLocalization.Name.IncomingChangesViewTitle),
                    wasIncomingChangesSelected,
                    mTabButtonWidth);

            if (isIncomingChangesSelected)
                ShowIncomingChangesView();
        }

        void ChangesetsTabButtonGUI()
        {
            bool wasChangesetsSelected =
                IsViewSelected(SelectedTab.Changesets);

            bool isChangesetsSelected = mChangesetsTabButton.
                DrawTabButton(
                    PlasticLocalization.GetString(PlasticLocalization.Name.ChangesetsViewTitle),
                    wasChangesetsSelected,
                    mTabButtonWidth);

            if (isChangesetsSelected)
                ShowChangesetsView();
        }

        void ShelvesTabButtonGUI()
        {
            if (ShelvesTab == null)
            {
                DrawStaticElement.Empty();
                return;
            }

            bool wasShelvesSelected =
                 IsViewSelected(SelectedTab.Shelves);

            bool isCloseButtonClicked;

            bool isShelvesSelected = mShelvesTabButton.
                DrawClosableTabButton(PlasticLocalization.GetString(
                    PlasticLocalization.Name.ShelvesViewTitle),
                    wasShelvesSelected,
                    true,
                    mTabButtonWidth,
                    mParentWindow.Repaint,
                    out isCloseButtonClicked);

            if (isCloseButtonClicked)
            {
                CloseShelvesTab();
                return;
            }

            if (isShelvesSelected)
                SetSelectedView(SelectedTab.Shelves);
        }

        void BranchesTabButtonGUI()
        {
            if (BranchesTab == null)
            {
                DrawStaticElement.Empty();
                return;
            }

            bool wasBranchesSelected =
                 IsViewSelected(SelectedTab.Branches);

            bool isCloseButtonClicked;

            bool isBranchesSelected = mBranchesTabButton.
                DrawClosableTabButton(PlasticLocalization.GetString(
                    PlasticLocalization.Name.BranchesViewTitle),
                    wasBranchesSelected,
                    true,
                    mTabButtonWidth,
                    mParentWindow.Repaint,
                    out isCloseButtonClicked);

            if (isCloseButtonClicked)
            {
                CloseBranchesTab();
                return;
            }

            if (isBranchesSelected)
                SetSelectedView(SelectedTab.Branches);
        }

        void LocksTabButtonGUI()
        {
            if (LocksTab == null)
            {
                DrawStaticElement.Empty();
                return;
            }

            var wasLocksTabSelected = IsViewSelected(SelectedTab.Locks);

            bool isCloseButtonClicked;

            var isLocksTabSelected = mLocksTabButton.DrawClosableTabButton(
                PlasticLocalization.Name.LocksViewTitle.GetString(),
                wasLocksTabSelected,
                true,
                mTabButtonWidth,
                mParentWindow.Repaint,
                out isCloseButtonClicked);

            if (isCloseButtonClicked)
            {
                CloseLocksTab();
                return;
            }

            if (isLocksTabSelected)
            {
                SetSelectedView(SelectedTab.Locks);
            }
        }

        void MergeTabButtonGUI()
        {
            if (MergeTab == null)
            {
                DrawStaticElement.Empty();
                return;
            }

            bool wasMergeSelected =
                IsViewSelected(SelectedTab.Merge);

            bool isCloseButtonClicked;

            bool isMergeSelected = mMergeTabButton.
                DrawClosableTabButton(
                    PlasticLocalization.Name.MainSidebarMergeItem.GetString(),
                    wasMergeSelected,
                    true,
                    mTabButtonWidth,
                    mParentWindow.Repaint,
                    out isCloseButtonClicked);

            if (isCloseButtonClicked)
            {
                CloseMergeTab();
                ShowPendingChangesView();
                return;
            }

            if (isMergeSelected)
                ShowMergeView();
        }

        void HistoryTabButtonGUI()
        {
            if (HistoryTab == null)
            {
                DrawStaticElement.Empty();
                return;
            }

            bool wasHistorySelected =
                IsViewSelected(SelectedTab.History);

            bool isCloseButtonClicked;

            bool isHistorySelected = mHistoryTabButton.
                DrawClosableTabButton(
                    PlasticLocalization.GetString(PlasticLocalization.Name.History),
                    wasHistorySelected,
                    true,
                    mTabButtonWidth,
                    mParentWindow.Repaint,
                    out isCloseButtonClicked);

            if (isCloseButtonClicked)
            {
                CloseHistoryTab();
                return;
            }

            if (isHistorySelected)
                SetSelectedView(SelectedTab.History);
        }

        static bool ShouldRefreshView(
            bool isViewInitialized,
            bool hasObjectToSelect,
            bool isViewSelected)
        {
            // If the view is not initialized, it will be refreshed
            // during the initialization. So we don't need to refresh.
            if (!isViewInitialized)
                return false;

            // If there is an object to select, we should refresh the view.
            if (hasObjectToSelect)
                return true;

            // If the view is selected, no need to refresh.
            if (isViewSelected)
                return false;

            // Otherwise, refresh the view.
            return true;
        }

        float mTabButtonWidth = -1;

        ViewSwitcherState mState;

        TabButton mPendingChangesTabButton;
        TabButton mIncomingChangesTabButton;
        TabButton mChangesetsTabButton;
        TabButton mShelvesTabButton;
        TabButton mBranchesTabButton;
        TabButton mLocksTabButton;
        TabButton mMergeTabButton;
        TabButton mHistoryTabButton;

        StatusBar.IIncomingChangesNotification mIncomingChangesNotification;
        GluonNewIncomingChangesUpdater mGluonNewIncomingChangesUpdater;
        NewIncomingChangesUpdater mDeveloperNewIncomingChangesUpdater;

        CheckShelvedChanges.IUpdateShelvedChangesNotification mUpdateShelvedChanges;
        ShelvedChangesUpdater mShelvedChangesUpdater;

        WorkspaceWindow mWorkspaceWindow;

        readonly EditorWindow mParentWindow;
        readonly StatusBar mStatusBar;
        readonly ISaveAssets mSaveAssets;
        readonly WorkspaceOperationsMonitor mWorkspaceOperationsMonitor;
        readonly LaunchTool.IProcessExecutor mProcessExecutor;
        readonly LaunchTool.IShowDownloadPlasticExeWindow mShowDownloadPlasticExeWindow;
        readonly IAssetStatusCache mAssetStatusCache;
        readonly bool mIsGluonMode;
        readonly ViewHost mViewHost;
        readonly WorkspaceInfo mWkInfo;
        readonly RepositorySpec mRepSpec;
    }
}
