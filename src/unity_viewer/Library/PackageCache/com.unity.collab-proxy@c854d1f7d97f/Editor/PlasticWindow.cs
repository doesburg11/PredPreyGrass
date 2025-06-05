using System;
using System.Threading;
using System.Threading.Tasks;

using UnityEditor;
using UnityEngine;

using Codice.Client.BaseCommands;
using Codice.Client.Common;
using Codice.Client.Common.EventTracking;
using Codice.Client.Common.Threading;
using Codice.CM.Common;
using Codice.LogWrapper;
using CodiceApp.EventTracking.Plastic;
using CodiceApp.EventTracking;
using GluonGui;
using PlasticGui;
using PlasticGui.WorkspaceWindow;
using PlasticGui.WorkspaceWindow.Merge;
using PlasticGui.WorkspaceWindow.NotificationBar;
using Unity.PlasticSCM.Editor.AssetMenu;
using Unity.PlasticSCM.Editor.AssetUtils;
using Unity.PlasticSCM.Editor.Configuration;
using Unity.PlasticSCM.Editor.Configuration.CloudEdition.Welcome;
using Unity.PlasticSCM.Editor.Developer;
using Unity.PlasticSCM.Editor.Inspector;
using Unity.PlasticSCM.Editor.Settings;
using Unity.PlasticSCM.Editor.Tool;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.Avatar;
using Unity.PlasticSCM.Editor.UI.Progress;
using Unity.PlasticSCM.Editor.UI.StatusBar;
using Unity.PlasticSCM.Editor.Views.CreateWorkspace;
using Unity.PlasticSCM.Editor.Views.Welcome;
using Unity.PlasticSCM.Editor.WebApi;

using GluonCheckIncomingChanges = PlasticGui.Gluon.WorkspaceWindow.CheckIncomingChanges;
using GluonNewIncomingChangesUpdater = PlasticGui.Gluon.WorkspaceWindow.NewIncomingChangesUpdater;
using GluonShelvedChangesNotification = Unity.PlasticSCM.Editor.Gluon.ShelvedChangesNotification;
using PlasticAssetModificationProcessor = Unity.PlasticSCM.Editor.AssetUtils.Processor.AssetModificationProcessor;
using ShelvedChangesNotification = Unity.PlasticSCM.Editor.Developer.ShelvedChangesNotification;

namespace Unity.PlasticSCM.Editor
{
    internal class PlasticWindow : EditorWindow,
        CheckIncomingChanges.IAutoRefreshIncomingChangesView,
        GluonCheckIncomingChanges.IAutoRefreshIncomingChangesView,
        CheckShelvedChanges.IAutoRefreshApplyShelveView,
        CreateWorkspaceView.ICreateWorkspaceListener
    {
        internal bool ShowWelcomeViewForTesting { get; set; }

        internal WelcomeView WelcomeViewForTesting { get { return mWelcomeView; } }

        internal WorkspaceWindow WorkspaceWindowForTesting { get { return mWorkspaceWindow; } }

        internal ViewSwitcher ViewSwitcherForTesting { get { return mViewSwitcher; } }

        internal ViewHost ViewHostForTesting { get { return mViewHost; } }

        internal CmConnection CmConnectionForTesting { get { return CmConnection.Get(); } }

        internal IShelvedChangesUpdater ShelvedChangesUpdater { get { return mShelvedChangesUpdater; } }

        internal WelcomeView GetWelcomeView()
        {
            if (mWelcomeView != null)
                return mWelcomeView;

            mWelcomeView = new WelcomeView(
                this,
                this,
                PlasticGui.Plastic.API,
                PlasticGui.Plastic.WebRestAPI);

            return mWelcomeView;
        }

        internal PendingChangesOptionsFoldout.IAutoRefreshView GetPendingChangesView()
        {
            return mViewSwitcher != null ? mViewSwitcher.PendingChangesTab : null;
        }

        internal void UpdateWindowIcon(PlasticNotification.Status status)
        {
            Texture windowIcon = PlasticNotification.GetIcon(status);

            if (titleContent.image != windowIcon)
                titleContent.image = windowIcon;
        }

        internal void RefreshWorkspaceUI()
        {
            InitializePlastic();
            Repaint();

            OnFocus();
        }

        internal void InitializePlastic()
        {
            if (mForceToReOpen)
            {
                mForceToReOpen = false;
                return;
            }

            try
            {
                if (UnityConfigurationChecker.NeedsConfiguration() ||
                    ShowWelcomeViewForTesting)
                    return;

                mWkInfo = FindWorkspace.InfoForApplicationPath(
                    ApplicationDataPath.Get(), PlasticGui.Plastic.API);

                if (mWkInfo == null)
                    return;

                PlasticPlugin.EnableForWorkspace(mWkInfo);

                // PlasticPlugin.EnableForWorkspace may trigger a workspace metadata
                // upgrade that modifies the repSpec. So, we need to calculate the repSpec
                // after calling it to ensure it is up-to-date.
                mRepSpec = PlasticGui.Plastic.API.GetRepositorySpec(mWkInfo);

                DisableVCSIfEnabled(mWkInfo.ClientPath);

                mIsGluonMode = PlasticGui.Plastic.API.IsGluonWorkspace(mWkInfo);

                mViewHost = new ViewHost();

                mStatusBar = new StatusBar();

                mSaveAssets = new SaveAssets();

                mViewSwitcher = new ViewSwitcher(
                    mRepSpec,
                    mWkInfo,
                    mViewHost,
                    mIsGluonMode,
                    PlasticPlugin.AssetStatusCache,
                    mShowDownloadPlasticExeWindow,
                    mProcessExecutor,
                    PlasticPlugin.WorkspaceOperationsMonitor,
                    mSaveAssets,
                    mStatusBar,
                    this);

                InitializeNewIncomingChanges(
                    mWkInfo, mViewSwitcher, mIsGluonMode);

                InitializeShelvedChanges(
                    mWkInfo,
                    mRepSpec,
                    mViewSwitcher,
                    mShowDownloadPlasticExeWindow,
                    mIsGluonMode);

                // Create a CooldownWindowDelayer to make the auto-refresh changes delayed.
                // In this way, we cover the following scenario:
                // * When Unity Editor window is activated it writes some files to its Temp
                //   folder. This causes the fswatcher to process those events.
                // * We need to wait until the fswatcher finishes processing the events,
                //   otherwise the NewChangesInWk method will return TRUE because there
                //   are pending events to process, which causes an unwanted 'get pending
                //   changes' operation when there are no new changes.
                // * So, we need to delay the auto-refresh call in order
                //   to give the fswatcher enough time to process the events.
                mCooldownAutoRefreshChangesAction = new CooldownWindowDelayer(
                    () =>
                    {
                        mViewSwitcher.AutoRefreshPendingChangesView();
                        mViewSwitcher.AutoRefreshIncomingChangesView();
                    },
                    UnityConstants.AUTO_REFRESH_CHANGES_DELAYED_INTERVAL);

                mWorkspaceWindow = new WorkspaceWindow(
                    mWkInfo,
                    mViewHost,
                    mViewSwitcher,
                    mStatusBar,
                    mViewSwitcher,
                    mDeveloperNewIncomingChangesUpdater,
                    mShelvedChangesUpdater,
                    this);

                mViewSwitcher.SetWorkspaceWindow(mWorkspaceWindow);

                mStatusBar.Initialize(
                    mWorkspaceWindow,
                    mIncomingChangesNotification,
                    mShelvedChangesNotification);

                mViewSwitcher.InitializeFromState(mViewSwitcherState);

                PlasticApp.RegisterWorkspaceWindow(mWorkspaceWindow);
                PlasticPlugin.WorkspaceOperationsMonitor.RegisterWindow(
                    mWorkspaceWindow,
                    mViewHost,
                    mDeveloperNewIncomingChangesUpdater);

                UnityStyles.Initialize(Repaint);

                AssetMenuItems.BuildOperations(
                    mWkInfo,
                    PlasticGui.Plastic.API,
                    mWorkspaceWindow,
                    mViewSwitcher,
                    mViewSwitcher,
                    mViewHost,
                    PlasticPlugin.WorkspaceOperationsMonitor,
                    mSaveAssets,
                    mDeveloperNewIncomingChangesUpdater,
                    mShelvedChangesUpdater,
                    PlasticPlugin.AssetStatusCache,
                    mViewSwitcher,
                    mViewSwitcher,
                    mShowDownloadPlasticExeWindow,
                    mIsGluonMode);

                DrawInspectorOperations.BuildOperations(
                    mWkInfo,
                    PlasticGui.Plastic.API,
                    mWorkspaceWindow,
                    mViewSwitcher,
                    mViewSwitcher,
                    mViewHost,
                    PlasticPlugin.WorkspaceOperationsMonitor,
                    mSaveAssets,
                    mDeveloperNewIncomingChangesUpdater,
                    mShelvedChangesUpdater,
                    PlasticPlugin.AssetStatusCache,
                    mViewSwitcher,
                    mViewSwitcher,
                    mShowDownloadPlasticExeWindow,
                    mIsGluonMode);

                mLastUpdateTime = EditorApplication.timeSinceStartup;

                mViewSwitcher.ShowBranchesViewIfNeeded();
                mViewSwitcher.ShowShelvesViewIfNeeded();
                mViewSwitcher.ShowLocksViewIfNeeded();
                MergeInProgress.ShowIfNeeded(mWkInfo, mViewSwitcher);

                // Note: this need to be initialized regardless of the type of the UVCS Edition installed
                InitializeCloudSubscriptionData();
                InitializeCurrentUser();

                if (!EditionToken.IsCloudEdition())
                    return;

                InitializeNotificationBarUpdater(
                    mWkInfo, mStatusBar.NotificationBar);
            }
            catch (Exception ex)
            {
                mException = ex;

                ExceptionsHandler.HandleException("InitializePlastic", ex);
            }
        }

        void CheckIncomingChanges.IAutoRefreshIncomingChangesView.IfVisible()
        {
            mViewSwitcher.AutoRefreshIncomingChangesView();
        }

        void GluonCheckIncomingChanges.IAutoRefreshIncomingChangesView.IfVisible()
        {
            mViewSwitcher.AutoRefreshIncomingChangesView();
        }

        void CheckShelvedChanges.IAutoRefreshApplyShelveView.IfVisible()
        {
            mViewSwitcher.AutoRefreshMergeView();
        }

        void CreateWorkspaceView.ICreateWorkspaceListener.OnWorkspaceCreated(
            WorkspaceInfo wkInfo, bool isGluonMode)
        {
            mWkInfo = wkInfo;
            mRepSpec = PlasticGui.Plastic.API.GetRepositorySpec(wkInfo);
            mIsGluonMode = isGluonMode;
            mWelcomeView = null;

            PlasticPlugin.Enable();

            if (mIsGluonMode)
                ConfigurePartialWorkspace.AsFullyChecked(mWkInfo);

            InitializePlastic();
            Repaint();
        }

        void OnEnable()
        {
            // Note: this log isn't visible if the window is opened automatically at startup,
            // as the logs are not initialized yet (later, conditionally, in PlasticPlugin.Enable())
            mLog.Debug("OnEnable");

            wantsMouseMove = true;

            if (mException != null)
                return;

            minSize = new Vector2(
                UnityConstants.PLASTIC_WINDOW_MIN_SIZE_WIDTH,
                UnityConstants.PLASTIC_WINDOW_MIN_SIZE_HEIGHT);

            UpdateWindowIcon(PlasticNotification.Status.None);

            RegisterApplicationFocusHandlers(this);

            if (!PlasticPlugin.ConnectionMonitor.IsConnected)
                return;

            PlasticPlugin.Enable();

            InitializePlastic();
        }

        void OnDisable()
        {
            mLog.Debug("OnDisable");

            // We need to disable MonoFSWatcher because otherwise it hangs
            // when you move the window between monitors with different scale
            PlasticApp.DisableMonoFsWatcherIfNeeded();

            if (mException != null)
                return;

            UnRegisterApplicationFocusHandlers(this);

            ClosePlastic(this);
        }

        void OnDestroy()
        {
            mLog.Debug("OnDestroy");

            if (mException != null)
                return;

            if (mWkInfo == null)
                return;

            if (!PlasticApp.HasRunningOperation())
                return;

            bool bCloseWindow = GuiMessage.ShowQuestion(
                PlasticLocalization.GetString(PlasticLocalization.Name.OperationRunning),
                PlasticLocalization.GetString(PlasticLocalization.Name.ConfirmClosingRunningOperation),
                PlasticLocalization.GetString(PlasticLocalization.Name.YesButton));

            if (bCloseWindow)
                return;

            mLog.Debug(
                "Show window again because the user doesn't want " +
                "to quit it due to there is an operation running");

            mForceToReOpen = true;

            ReOpenPlasticWindow(this);
        }

        void OnFocus()
        {
            mLog.Debug("OnFocus");

            if (mException != null)
                return;

            if (mWkInfo == null)
                return;

            if (!PlasticPlugin.ConnectionMonitor.IsConnected)
                return;

            // We don't want to auto-refresh the views when the window
            // is focused due to a right mouse button click because
            // if there is no internet connection a dialog appears and
            // it prevents being able to open the context menu in order
            // to close the Plastic SCM window
            if (Mouse.IsRightMouseButtonPressed(Event.current))
                return;

            mCooldownAutoRefreshChangesAction.Ping();
        }

        void OnGUI()
        {
            if (!PlasticPlugin.ConnectionMonitor.IsConnected)
            {
                DoNotConnectedArea();
                return;
            }

            if (mException != null)
            {
                DoExceptionErrorArea();
                return;
            }

            try
            {
                bool clientNeedsConfiguration = UnityConfigurationChecker.NeedsConfiguration() || ShowWelcomeViewForTesting;

                WelcomeView welcomeView = GetWelcomeView();

                if (clientNeedsConfiguration && welcomeView.autoLoginState == AutoLogin.State.Off)
                {
                    welcomeView.autoLoginState = AutoLogin.State.Started;
                }

                if (NeedsToDisplayWelcomeView(clientNeedsConfiguration, mWkInfo))
                {
                    welcomeView.OnGUI(clientNeedsConfiguration);
                    return;
                }

                //TODO: Codice - beta: hide the switcher until the update dialog is implemented
                //DrawGuiModeSwitcher.ForMode(
                //    isGluonMode, plasticClient, changesTreeView, editorWindow);

                DoTabToolbar(
                    mWkInfo,
                    mRepSpec,
                    mViewSwitcher,
                    mShowDownloadPlasticExeWindow,
                    mProcessExecutor,
                    mIsGluonMode,
                    mIsCloudOrganization,
                    mIsUnityOrganization,
                    mIsUGOSubscription);

                mViewSwitcher.TabViewGUI(GetCurrentUser());

                if (mWorkspaceWindow.IsOperationInProgress())
                    DrawProgressForOperations.For(
                        mWorkspaceWindow, mWorkspaceWindow.Progress,
                        position.width);

                mStatusBar.OnGUI();
            }
            catch (Exception ex)
            {
                if (IsExitGUIException(ex))
                    throw;

                GUI.enabled = true;

                if (IsIMGUIPaintException(ex))
                {
                    ExceptionsHandler.LogException("PlasticWindow", ex);
                    return;
                }

                mException = ex;

                DoExceptionErrorArea();

                ExceptionsHandler.HandleException("OnGUI", ex);
            }
        }

        void Update()
        {
            if (mException != null)
                return;

            if (mWkInfo == null)
                return;

            try
            {
                double currentUpdateTime = EditorApplication.timeSinceStartup;
                double elapsedSeconds = currentUpdateTime - mLastUpdateTime;

                mViewSwitcher.Update();
                mWorkspaceWindow.OnParentUpdated(elapsedSeconds);

                if (mWelcomeView != null)
                    mWelcomeView.Update();

                mLastUpdateTime = currentUpdateTime;
            }
            catch (Exception ex)
            {
                mException = ex;

                ExceptionsHandler.HandleException("Update", ex);
            }
        }

        void OnApplicationActivated()
        {
            mLog.Debug("OnApplicationActivated");

            if (mException != null)
                return;

            if (!PlasticPlugin.ConnectionMonitor.IsConnected)
                return;

            if (UnityConfigurationChecker.NeedsConfiguration() ||
                ShowWelcomeViewForTesting)
                return;

            Reload.IfWorkspaceConfigChanged(
                PlasticGui.Plastic.API, mWkInfo, mIsGluonMode,
                ExecuteFullReload);

            if (mWkInfo == null)
                return;

            NewIncomingChanges.LaunchUpdater(
                mDeveloperNewIncomingChangesUpdater,
                mGluonNewIncomingChangesUpdater);

            mShelvedChangesUpdater.Start();
            mShelvedChangesUpdater.Update(DateTime.Now);

            if (!PlasticApp.HasRunningOperation())
                mCooldownAutoRefreshChangesAction.Ping();

            ((IWorkspaceWindow)mWorkspaceWindow).UpdateTitle();
        }

        void OnApplicationDeactivated()
        {
            mLog.Debug("OnApplicationDeactivated");

            if (mException != null)
                return;

            if (mWkInfo == null)
                return;

            if (!PlasticPlugin.ConnectionMonitor.IsConnected)
                return;

            NewIncomingChanges.StopUpdater(
                mDeveloperNewIncomingChangesUpdater,
                mGluonNewIncomingChangesUpdater);

            mShelvedChangesUpdater.Stop();
        }

        void ExecuteFullReload()
        {
            mException = null;

            ClosePlastic(this);

            InitializePlastic();
        }

        void InitializeCloudSubscriptionData()
        {
            mIsCloudOrganization = false;
            mIsUnityOrganization = false;
            mIsUGOSubscription = false;

            if (mRepSpec == null)
                return;

            mIsCloudOrganization = PlasticGui.Plastic.API.IsCloud(mRepSpec.Server);

            if (!mIsCloudOrganization)
                return;

            mIsUnityOrganization = OrganizationsInformation.IsUnityOrganization(mRepSpec.Server);

            string organizationName = ServerOrganizationParser.GetOrganizationFromServer(mRepSpec.Server);

            Task.Run(
                () =>
                {
                    string authToken = AuthToken.GetForServer(mRepSpec.Server);

                    if (string.IsNullOrEmpty(authToken))
                        return null;

                    return WebRestApiClient.PlasticScm.GetSubscriptionDetails(
                        organizationName, authToken);
                }).ContinueWith(
                t =>
                {
                    if (t.Result == null)
                    {
                        mLog.DebugFormat(
                            "Error getting Subscription details for organization {0}",
                            organizationName);
                        return;
                    }

                    mIsUGOSubscription = t.Result.OrderSource == UGO_ORDER_SOURCE;
                });
        }

        void InitializeCurrentUser()
        {
            PlasticThreadPool.Run(new WaitCallback(delegate
            {
                try
                {
                    SetCurrentUser(PlasticGui.Plastic.
                        API.GetCurrentUser(mRepSpec.Server));
                }
                catch (Exception ex)
                {
                    mLog.ErrorFormat("Error loading the current user: {0}", ex.Message);
                    mLog.DebugFormat("Stack trace: {0}", ex.StackTrace);
                }
            }));
        }

        void DoNotConnectedArea()
        {
            string labelText = PlasticLocalization.GetString(
                PlasticLocalization.Name.NotConnectedTryingToReconnect);

            string buttonText = PlasticLocalization.GetString(
                PlasticLocalization.Name.TryNowButton);

            GUI.enabled = !PlasticPlugin.ConnectionMonitor.IsTryingReconnection;

            DrawActionHelpBox.For(
                Images.GetInfoDialogIcon(), labelText, buttonText,
                PlasticPlugin.ConnectionMonitor.CheckConnection);

            GUI.enabled = true;
        }

        void DoExceptionErrorArea()
        {
            string labelText = PlasticLocalization.GetString(
                PlasticLocalization.Name.UnexpectedError);

            string buttonText = PlasticLocalization.GetString(
                PlasticLocalization.Name.ReloadButton);

            DrawActionHelpBox.For(
                Images.GetErrorDialogIcon(), labelText, buttonText,
                ExecuteFullReload);
        }

        void InitializeNewIncomingChanges(
            WorkspaceInfo wkInfo,
            ViewSwitcher viewSwitcher,
            bool bIsGluonMode)
        {
            if (bIsGluonMode)
            {
                Gluon.IncomingChangesNotification gluonNotification =
                    new Gluon.IncomingChangesNotification(wkInfo, viewSwitcher, this);
                mGluonNewIncomingChangesUpdater =
                    NewIncomingChanges.BuildUpdaterForGluon(
                        wkInfo, viewSwitcher, gluonNotification, this, gluonNotification,
                        new GluonCheckIncomingChanges.CalculateIncomingChanges());
                mIncomingChangesNotification = gluonNotification;
                return;
            }

            IncomingChangesNotification developerNotification =
                new IncomingChangesNotification(wkInfo, viewSwitcher, this);
            mDeveloperNewIncomingChangesUpdater =
                NewIncomingChanges.BuildUpdaterForDeveloper(
                    wkInfo, viewSwitcher, developerNotification,
                    this, developerNotification);
            mIncomingChangesNotification = developerNotification;
        }

        void InitializeShelvedChanges(
            WorkspaceInfo wkInfo,
            RepositorySpec repSpec,
            ViewSwitcher viewSwitcher,
            LaunchTool.IShowDownloadPlasticExeWindow showDownloadPlasticExeWindow,
            bool bIsGluonMode)
        {
            mShelvedChangesNotification = bIsGluonMode ?
                new GluonShelvedChangesNotification(
                    wkInfo,
                    repSpec,
                    viewSwitcher,
                    showDownloadPlasticExeWindow,
                    this) :
                new ShelvedChangesNotification(
                    wkInfo,
                    repSpec,
                    viewSwitcher,
                    this) as StatusBar.IShelvedChangesNotification;

            mShelvedChangesUpdater = new ShelvedChangesUpdater(
                wkInfo,
                new UnityPlasticTimerBuilder(),
                this,
                new CalculateShelvedChanges(new BaseCommandsImpl()),
                mShelvedChangesNotification);

            viewSwitcher.SetShelvedChanges(mShelvedChangesUpdater, mShelvedChangesNotification);
            mShelvedChangesNotification.SetShelvedChangesUpdater(mShelvedChangesUpdater);

            mShelvedChangesUpdater.Start();
        }

        void InitializeNotificationBarUpdater(
            WorkspaceInfo wkInfo,
            INotificationBar notificationBar)
        {
            mNotificationBarUpdater = new NotificationBarUpdater(
                notificationBar,
                PlasticGui.Plastic.WebRestAPI,
                new UnityPlasticTimerBuilder(),
                new NotificationBarUpdater.NotificationBarConfig(),
                BuildEventModel.CurrentApplicationString,
                UVCPackageVersion.Value,
                BuildEvent.CurrentPlatform.ToString());
            mNotificationBarUpdater.Start();
            mNotificationBarUpdater.SetWorkspace(wkInfo);
        }

        void SetCurrentUser(ResolvedUser currentUser)
        {
            lock (mCurrentUserLock)
            {
                mCurrentUser = currentUser;
            }
        }

        ResolvedUser GetCurrentUser()
        {
            lock (mCurrentUserLock)
            {
                return mCurrentUser;
            }
        }

        static void DoTabToolbar(
            WorkspaceInfo workspaceInfo,
            RepositorySpec repSpec,
            ViewSwitcher viewSwitcher,
            LaunchTool.IShowDownloadPlasticExeWindow showDownloadPlasticExeWindow,
            LaunchTool.IProcessExecutor processExecutor,
            bool isGluonMode,
            bool isCloudOrganization,
            bool isUnityOrganization,
            bool isUGOSubscription)
        {
            EditorGUILayout.BeginHorizontal(EditorStyles.toolbar);

            viewSwitcher.TabButtonsGUI();

            GUILayout.FlexibleSpace();

            EditorGUILayout.BeginHorizontal(EditorStyles.toolbar);

            GUILayout.Space(2);

            DoSearchField(viewSwitcher);

            GUILayout.Space(2);

            EditorGUILayout.EndHorizontal();

            EditorGUILayout.BeginHorizontal(EditorStyles.toolbar);

            DoToolbarButtons(
                workspaceInfo,
                repSpec,
                viewSwitcher,
                showDownloadPlasticExeWindow,
                processExecutor,
                isGluonMode,
                isCloudOrganization,
                isUnityOrganization,
                isUGOSubscription);

            EditorGUILayout.EndHorizontal();

            EditorGUILayout.EndHorizontal();
        }

        static void DoSearchField(ViewSwitcher viewSwitcher)
        {
            if (viewSwitcher.IsViewSelected(ViewSwitcher.SelectedTab.PendingChanges))
            {
                viewSwitcher.PendingChangesTab.DrawSearchFieldForTab();
                return;
            }

            if (viewSwitcher.IsViewSelected(ViewSwitcher.SelectedTab.IncomingChanges))
            {
                viewSwitcher.IncomingChangesTab.DrawSearchFieldForTab();
                return;
            }

            if (viewSwitcher.IsViewSelected(ViewSwitcher.SelectedTab.Changesets))
            {
                viewSwitcher.ChangesetsTab.DrawSearchFieldForTab();
                return;
            }

            if (viewSwitcher.IsViewSelected(ViewSwitcher.SelectedTab.Branches))
            {
                viewSwitcher.BranchesTab.DrawSearchFieldForTab();
                return;
            }

            if (viewSwitcher.IsViewSelected(ViewSwitcher.SelectedTab.Shelves))
            {
                viewSwitcher.ShelvesTab.DrawSearchFieldForTab();
                return;
            }

            if (viewSwitcher.IsViewSelected(ViewSwitcher.SelectedTab.Locks))
            {
                viewSwitcher.LocksTab.DrawSearchFieldForTab();
                return;
            }

            if (viewSwitcher.IsViewSelected(ViewSwitcher.SelectedTab.Merge))
            {
                viewSwitcher.MergeTab.DrawSearchFieldForTab();
                return;
            }

            if (viewSwitcher.IsViewSelected(ViewSwitcher.SelectedTab.History))
            {
                viewSwitcher.HistoryTab.DrawSearchFieldForTab();
                return;
            }
        }

        static void DoToolbarButtons(
            WorkspaceInfo wkInfo,
            RepositorySpec repSpec,
            ViewSwitcher viewSwitcher,
            LaunchTool.IShowDownloadPlasticExeWindow showDownloadPlasticExeWindow,
            LaunchTool.IProcessExecutor processExecutor,
            bool isGluonMode,
            bool isCloudOrganization,
            bool isUnityOrganization,
            bool isUGOSubscription)
        {
            if (viewSwitcher.IsViewSelected(ViewSwitcher.SelectedTab.Changesets))
            {
                viewSwitcher.ChangesetsTab.DrawDateFilter();
            }
            else if (viewSwitcher.IsViewSelected(ViewSwitcher.SelectedTab.Branches))
            {
                viewSwitcher.BranchesTab.DrawShowHiddenBranchesButton();
                viewSwitcher.BranchesTab.DrawDateFilter();
            }
            else
            {
                DrawStaticElement.Empty();
            }

            if (viewSwitcher.IsViewSelected(ViewSwitcher.SelectedTab.Shelves))
            {
                viewSwitcher.ShelvesTab.DrawOwnerFilter();
            }

            if (DrawToolbarButton(
                    Images.GetRefreshIcon(),
                    PlasticLocalization.Name.RefreshButton.GetString()))
            {
                viewSwitcher.RefreshSelectedView();
            }

            if (isGluonMode)
            {
                if (DrawActionButton.For(PlasticLocalization.Name.Configure.GetString()))
                {
                    LaunchTool.OpenWorkspaceConfiguration(
                        showDownloadPlasticExeWindow, processExecutor, wkInfo, isGluonMode);
                }
            }
            else
            {
                DrawStaticElement.Empty();
            }

            if (DrawToolbarButton(
                    Images.GetShelveIcon(),
                    PlasticLocalization.Name.ShowShelvesButton.GetString()))
            {
                TrackFeatureUseEvent.For(
                    repSpec,
                    TrackFeatureUseEvent.Features.UnityPackage.ShowShelvesViewFromToolbarButton);

                viewSwitcher.ShowShelvesView();
            }

            if (DrawToolbarButton(
                    Images.GetBranchesIcon(),
                    PlasticLocalization.Name.Branches.GetString()))
            {
                viewSwitcher.ShowBranchesView();
            }

            if (!isGluonMode)
            {
                if (DrawToolbarButton(
                        Images.GetBranchExplorerIcon(),
                        PlasticLocalization.Name.BranchExplorerMenu.GetString()))
                {
                    LaunchTool.OpenBranchExplorer(
                        showDownloadPlasticExeWindow, processExecutor, wkInfo, isGluonMode);
                }
            }
            else
            {
                DrawStaticElement.Empty();
            }

            if (DrawToolbarButton(
                    Images.GetLockIcon(),
                    PlasticLocalization.Name.ShowLocks.GetString()))
            {
                viewSwitcher.ShowLocksView();
            }

            if (isCloudOrganization)
            {
                if (DrawToolbarButton(
                    Images.GetInviteUsersIcon(),
                    isUnityOrganization
                        ? PlasticLocalization.Name.InviteMembersToProject.GetString()
                        : PlasticLocalization.Name.InviteMembersToOrganization.GetString()))
                {
                    InviteMembers(repSpec);
                }
            }
            else
            {
                DrawStaticElement.Empty();
            }

            if (isCloudOrganization && isUGOSubscription)
            {
                if (DrawToolbarTextButton(PlasticLocalization.Name.UpgradePlan.GetString()))
                {
                    OpenDevOpsUpgradePlanUrl();
                }
            }
            else
            {
                DrawStaticElement.Empty();
            }

            //TODO: Add settings button tooltip localization
            if (DrawToolbarButton(Images.GetSettingsIcon(), string.Empty))
            {
                ShowSettingsContextMenu(
                    showDownloadPlasticExeWindow,
                    processExecutor,
                    wkInfo,
                    isGluonMode,
                    isCloudOrganization);
            }
        }

        static bool DrawToolbarButton(Texture icon, string tooltip)
        {
            return GUILayout.Button(
                new GUIContent(icon, tooltip),
                EditorStyles.toolbarButton,
                GUILayout.Width(26));
        }

        static bool DrawToolbarTextButton(string text)
        {
            return GUILayout.Button(
                new GUIContent(text, string.Empty),
                EditorStyles.toolbarButton);
        }

        static void InviteMembers(RepositorySpec repSpec)
        {
            string organizationName = ServerOrganizationParser.GetOrganizationFromServer(repSpec.Server);

            CurrentUserAdminCheckResponse response = null;

            IThreadWaiter waiter = ThreadWaiter.GetWaiter(50);
            waiter.Execute(
                /*threadOperationDelegate*/
                delegate
                {
                    string authToken = AuthToken.GetForServer(repSpec.Server);

                    if (string.IsNullOrEmpty(authToken))
                    {
                        return;
                    }

                    response = WebRestApiClient.PlasticScm.IsUserAdmin(organizationName, authToken);
                },
                /*afterOperationDelegate*/
                delegate
                {
                    if (waiter.Exception != null)
                    {
                        ExceptionsHandler.LogException("IsUserAdmin", waiter.Exception);

                        OpenUnityDashboardInviteUsersUrl(repSpec);
                        return;
                    }

                    if (response == null)
                    {
                        mLog.DebugFormat(
                            "Error checking if the user is the organization admin for {0}",
                            organizationName);

                        OpenUnityDashboardInviteUsersUrl(repSpec);
                        return;
                    }

                    if (response.Error != null)
                    {
                        mLog.DebugFormat(
                          "Error checking if the user is the organization admin: {0}",
                          string.Format("Unable to get IsUserAdminResponse: {0} [code {1}]",
                              response.Error.Message,
                              response.Error.ErrorCode));

                        OpenUnityDashboardInviteUsersUrl(repSpec);
                        return;
                    }

                    if (!response.IsCurrentUserAdmin)
                    {
                        GuiMessage.ShowInformation(
                            PlasticLocalization.GetString(PlasticLocalization.Name.InviteMembersTitle),
                            PlasticLocalization.GetString(PlasticLocalization.Name.InviteMembersToOrganizationNotAdminError));

                        return;
                    }

                    OpenUnityDashboardInviteUsersUrl(repSpec);
                });
        }

        static void OpenUnityDashboardInviteUsersUrl(RepositorySpec repSpec)
        {
            OpenInviteUsersPage.Run(repSpec, UnityUrl.UnityDashboard.UnityCloudRequestSource.Editor);
        }

        static void ShowSettingsContextMenu(
            LaunchTool.IShowDownloadPlasticExeWindow showDownloadPlasticExeWindow,
            LaunchTool.IProcessExecutor processExecutor,
            WorkspaceInfo wkInfo,
            bool isGluonMode,
            bool isCloudOrganization)
        {
            GenericMenu menu = new GenericMenu();

            string openToolText = isGluonMode ?
                PlasticLocalization.Name.OpenInGluon.GetString() :
                PlasticLocalization.Name.OpenInDesktopApp.GetString();

            menu.AddItem(
                new GUIContent(openToolText),
                false,
                () => LaunchTool.OpenGUIForMode(
                    showDownloadPlasticExeWindow,
                    processExecutor,
                    wkInfo,
                    isGluonMode));

            if (isCloudOrganization)
            {
                menu.AddItem(
                    new GUIContent(PlasticLocalization.Name.OpenInUnityCloud.GetString()),
                    false,
                    () => OpenUnityCloudRepository.Run(wkInfo));
            }

            menu.AddSeparator(string.Empty);

            menu.AddItem(
                new GUIContent(PlasticLocalization.Name.Settings.GetString()),
                false,
                OpenPlasticProjectSettings.ByDefault);

            menu.AddItem(
                new GUIContent(PlasticAssetModificationProcessor.IsManualCheckoutEnabled ?
                    PlasticLocalization.Name.DisableForcedCheckout.GetString() :
                    PlasticLocalization.Name.EnableForcedCheckout.GetString()),
                false,
                () => PlasticAssetModificationProcessor.SetManualCheckoutPreference(
                    !PlasticAssetModificationProcessor.IsManualCheckoutEnabled));

            menu.ShowAsContext();
        }

        static void OpenDevOpsUpgradePlanUrl()
        {
            Application.OpenURL(UnityUrl.DevOps.GetSignUp());
        }

        static void DisableVCSIfEnabled(string projectPath)
        {
            if (!VCSPlugin.IsEnabled())
                return;

            VCSPlugin.Disable();

            mLog.DebugFormat("Disabled VCS Plugin on Project: {0}",
                projectPath);
        }

        static void DisposeNewIncomingChanges(PlasticWindow window)
        {
            NewIncomingChanges.DisposeUpdater(
                window.mDeveloperNewIncomingChangesUpdater,
                window.mGluonNewIncomingChangesUpdater);

            window.mDeveloperNewIncomingChangesUpdater = null;
            window.mGluonNewIncomingChangesUpdater = null;
        }

        static void DisposeShelvedChanges(PlasticWindow window)
        {
            if (window.mShelvedChangesUpdater == null)
                return;

            window.mShelvedChangesUpdater.Dispose();
            window.mShelvedChangesUpdater = null;
        }

        static void DisposeNotificationBarUpdater(PlasticWindow window)
        {
            if (window.mNotificationBarUpdater == null)
                return;

            window.mNotificationBarUpdater.Dispose();
            window.mNotificationBarUpdater = null;
        }

        static void RegisterApplicationFocusHandlers(PlasticWindow window)
        {
            EditorWindowFocus.OnApplicationActivated += window.OnApplicationActivated;
            EditorWindowFocus.OnApplicationDeactivated += window.OnApplicationDeactivated;
        }

        static void UnRegisterApplicationFocusHandlers(PlasticWindow window)
        {
            EditorWindowFocus.OnApplicationActivated -= window.OnApplicationActivated;
            EditorWindowFocus.OnApplicationDeactivated -= window.OnApplicationDeactivated;
        }

        static void InitializePlasticOnForceToReOpen(PlasticWindow window)
        {
            if (window.mWkInfo == null)
                return;

            window.mViewSwitcher.OnEnable();

            window.InitializeNewIncomingChanges(
                window.mWkInfo,
                window.mViewSwitcher,
                window.mIsGluonMode);
            window.InitializeShelvedChanges(
                window.mWkInfo,
                window.mRepSpec,
                window.mViewSwitcher,
                window.mShowDownloadPlasticExeWindow,
                window.mIsGluonMode);

            PlasticApp.RegisterWorkspaceWindow(
                window.mWorkspaceWindow);

            if (PlasticPlugin.WorkspaceOperationsMonitor != null)
            {
                PlasticPlugin.WorkspaceOperationsMonitor.RegisterWindow(
                    window.mWorkspaceWindow,
                    window.mViewHost,
                    window.mDeveloperNewIncomingChangesUpdater);
            }

            if (!EditionToken.IsCloudEdition())
                return;

            window.InitializeNotificationBarUpdater(
                window.mWkInfo,
                window.mStatusBar.NotificationBar);
        }

        static void ClosePlastic(PlasticWindow window)
        {
            if (window.mViewSwitcher != null)
                window.mViewSwitcher.OnDisable();

            PlasticApp.UnRegisterWorkspaceWindow();

            if (PlasticPlugin.WorkspaceOperationsMonitor != null)
                PlasticPlugin.WorkspaceOperationsMonitor.UnRegisterWindow();

            DisposeNewIncomingChanges(window);
            DisposeShelvedChanges(window);

            DisposeNotificationBarUpdater(window);

            AvatarImages.Dispose();
        }

        static void ReOpenPlasticWindow(PlasticWindow closedWindow)
        {
            EditorWindow dockWindow = FindEditorWindow.ToDock<PlasticWindow>();

            PlasticWindow newWindow = InstantiateFrom(closedWindow);

            InitializePlasticOnForceToReOpen(newWindow);

            if (DockEditorWindow.IsAvailable())
                DockEditorWindow.To(dockWindow, newWindow);

            newWindow.Show();
            newWindow.Focus();
        }

        static bool NeedsToDisplayWelcomeView(
            bool clientNeedsConfiguration,
            WorkspaceInfo wkInfo)
        {
            if (clientNeedsConfiguration)
                return true;

            if (wkInfo == null)
                return true;

            return false;
        }

        static bool IsExitGUIException(Exception ex)
        {
            return ex is ExitGUIException;
        }

        static bool IsIMGUIPaintException(Exception ex)
        {
            if (!(ex is ArgumentException))
                return false;

            return ex.Message.StartsWith("Getting control") &&
                   ex.Message.Contains("controls when doing repaint");
        }

        static PlasticWindow InstantiateFrom(PlasticWindow window)
        {
            PlasticWindow result = Instantiate(window);
            result.mIsGluonMode = window.mIsGluonMode;
            result.mIsCloudOrganization = window.mIsCloudOrganization;
            result.mIsUnityOrganization = window.mIsUnityOrganization;
            result.mIsUGOSubscription = window.mIsUGOSubscription;
            result.mLastUpdateTime = window.mLastUpdateTime;
            result.mViewSwitcherState = window.mViewSwitcherState;
            result.mCurrentUser = window.mCurrentUser;
            result.mException = window.mException;
            result.mWkInfo = window.mWkInfo;
            result.mRepSpec = window.mRepSpec;
            result.mWelcomeView = window.mWelcomeView;
            result.mViewSwitcher = window.mViewSwitcher;
            result.mWorkspaceWindow = window.mWorkspaceWindow;
            result.mStatusBar = window.mStatusBar;
            result.mViewHost = window.mViewHost;
            result.mCooldownAutoRefreshChangesAction = window.mCooldownAutoRefreshChangesAction;
            result.mIncomingChangesNotification = window.mIncomingChangesNotification;
            result.mShelvedChangesNotification = window.mShelvedChangesNotification;
            result.mNotificationBarUpdater = window.mNotificationBarUpdater;
            result.mDeveloperNewIncomingChangesUpdater = window.mDeveloperNewIncomingChangesUpdater;
            result.mGluonNewIncomingChangesUpdater = window.mGluonNewIncomingChangesUpdater;
            result.mShelvedChangesUpdater = window.mShelvedChangesUpdater;
            result.mSaveAssets = window.mSaveAssets;
            return result;
        }

        static class Reload
        {
            internal static void IfWorkspaceConfigChanged(
                IPlasticAPI plasticApi,
                WorkspaceInfo lastWkInfo,
                bool lastIsGluonMode,
                Action reloadAction)
            {
                string applicationPath = ApplicationDataPath.Get();

                bool isGluonMode = false;
                WorkspaceInfo wkInfo = null;

                IThreadWaiter waiter = ThreadWaiter.GetWaiter(10);
                waiter.Execute(
                    /*threadOperationDelegate*/ delegate
                    {
                        wkInfo = FindWorkspace.
                            InfoForApplicationPath(applicationPath, plasticApi);

                        if (wkInfo != null)
                            isGluonMode = plasticApi.IsGluonWorkspace(wkInfo);
                    },
                    /*afterOperationDelegate*/ delegate
                    {
                        if (waiter.Exception != null)
                            return;

                        if (!IsWorkspaceConfigChanged(
                                lastWkInfo, wkInfo,
                                lastIsGluonMode, isGluonMode))
                            return;

                        reloadAction();
                    });
            }

            static bool IsWorkspaceConfigChanged(
                WorkspaceInfo lastWkInfo,
                WorkspaceInfo currentWkInfo,
                bool lastIsGluonMode,
                bool currentIsGluonMode)
            {
                if (lastIsGluonMode != currentIsGluonMode)
                    return true;

                if (lastWkInfo == null)
                    return currentWkInfo != null;

                return !lastWkInfo.Equals(currentWkInfo);
            }
        }

        [SerializeField]
        bool mForceToReOpen;
        bool mIsGluonMode;
        bool mIsCloudOrganization;
        bool mIsUnityOrganization;
        bool mIsUGOSubscription;
        double mLastUpdateTime = 0f;
        ViewSwitcherState mViewSwitcherState = new ViewSwitcherState();

        ResolvedUser mCurrentUser;

        Exception mException;
        object mCurrentUserLock = new object();
        WelcomeView mWelcomeView;
        WorkspaceWindow mWorkspaceWindow;
        StatusBar mStatusBar;

        [NonSerialized]
        WorkspaceInfo mWkInfo;
        RepositorySpec mRepSpec;
        ViewSwitcher mViewSwitcher;
        CooldownWindowDelayer mCooldownAutoRefreshChangesAction;
        StatusBar.IIncomingChangesNotification mIncomingChangesNotification;
        StatusBar.IShelvedChangesNotification mShelvedChangesNotification;
        ViewHost mViewHost;
        NotificationBarUpdater mNotificationBarUpdater;
        NewIncomingChangesUpdater mDeveloperNewIncomingChangesUpdater;
        GluonNewIncomingChangesUpdater mGluonNewIncomingChangesUpdater;
        ShelvedChangesUpdater mShelvedChangesUpdater;
        ISaveAssets mSaveAssets;

        LaunchTool.IShowDownloadPlasticExeWindow mShowDownloadPlasticExeWindow =
            new LaunchTool.ShowDownloadPlasticExeWindow();
        LaunchTool.IProcessExecutor mProcessExecutor =
            new LaunchTool.ProcessExecutor();

        const string UGO_ORDER_SOURCE = "UGO";

        static readonly ILog mLog = PlasticApp.GetLogger("PlasticWindow");
    }
}
