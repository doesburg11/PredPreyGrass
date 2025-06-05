using System;
using System.Collections.Generic;

using UnityEditor;
using UnityEditor.IMGUI.Controls;
using UnityEngine;

using Codice.Client.BaseCommands;
using Codice.Client.Commands;
using Codice.Client.Common;
using Codice.Client.Common.EventTracking;
using Codice.Client.Common.FsNodeReaders;
using Codice.Client.Common.Threading;
using Codice.CM.Common;
using Codice.CM.Common.Merge;
using Codice.CM.Common.Mount;
using Codice.LogWrapper;
using GluonGui;
using GluonGui.WorkspaceWindow.Views.Checkin.Operations;
using GluonGui.WorkspaceWindow.Views.Shelves;
using PlasticGui;
using PlasticGui.Help.Actions;
using PlasticGui.Help.Conditions;
using PlasticGui.WorkspaceWindow;
using PlasticGui.WorkspaceWindow.Diff;
using PlasticGui.WorkspaceWindow.Items;
using PlasticGui.WorkspaceWindow.Open;
using PlasticGui.WorkspaceWindow.PendingChanges;
using PlasticGui.WorkspaceWindow.PendingChanges.Changelists;
using Unity.PlasticSCM.Editor.AssetsOverlays.Cache;
using Unity.PlasticSCM.Editor.AssetUtils;
using Unity.PlasticSCM.Editor.AssetUtils.Processor;
using Unity.PlasticSCM.Editor.Gluon.Errors;
using Unity.PlasticSCM.Editor.Settings;
using Unity.PlasticSCM.Editor.Tool;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.Progress;
using Unity.PlasticSCM.Editor.UI.StatusBar;
using Unity.PlasticSCM.Editor.UI.Tree;
using Unity.PlasticSCM.Editor.Views.PendingChanges.Dialogs;
using Unity.PlasticSCM.Editor.Views.PendingChanges.PendingMergeLinks;
using Unity.PlasticSCM.Editor.Views.Changesets;

using GluonNewIncomingChangesUpdater = PlasticGui.Gluon.WorkspaceWindow.NewIncomingChangesUpdater;

namespace Unity.PlasticSCM.Editor.Views.PendingChanges
{
    internal partial class PendingChangesTab :
        IRefreshableView,
        PendingChangesOptionsFoldout.IAutoRefreshView,
        IPendingChangesView,
        CheckinUIOperation.ICheckinView,
        ShelveOperations.ICheckinView,
        PendingChangesViewPendingChangeMenu.IMetaMenuOperations,
        IPendingChangesMenuOperations,
        IChangelistMenuOperations,
        IOpenMenuOperations,
        PendingChangesViewPendingChangeMenu.IAdvancedUndoMenuOperations,
        IFilesFilterPatternsMenuOperations,
        PendingChangesViewMenu.IGetSelectedNodes,
        ChangesetsTab.IRevertToChangesetListener
    {

        internal bool ForceToShowComment { get { return mForceToShowComment; } }
        internal string EmptyStateMessage { get { return mEmptyStateData.Content.text; } }
        internal string CommentText { get { return mCommentText; } }
        internal IProgressControls ProgressControls { get { return mProgressControls; } }
        internal PendingChangesTreeView Table { get { return mPendingChangesTreeView; } }

        internal void SetMergeLinksForTesting(
            IDictionary<MountPoint, IList<PendingMergeLink>> mergeLinks)
        {
            mPendingMergeLinks = mergeLinks;

            UpdateMergeLinksList();
        }

        internal PendingChangesTab(
            WorkspaceInfo wkInfo,
            RepositorySpec repSpec,
            ViewHost viewHost,
            bool isGluonMode,
            WorkspaceWindow workspaceWindow,
            IViewSwitcher viewSwitcher,
            IShowChangesetInView showChangesetInView,
            IShowShelveInView showShelveInView,
            IMergeViewLauncher mergeViewLauncher,
            IHistoryViewLauncher historyViewLauncher,
            LaunchTool.IShowDownloadPlasticExeWindow showDownloadPlasticExeWindow,
            WorkspaceOperationsMonitor workspaceOperationsMonitor,
            ISaveAssets saveAssets,
            NewIncomingChangesUpdater developerNewIncomingChangesUpdater,
            GluonNewIncomingChangesUpdater gluonNewIncomingChangesUpdater,
            IAssetStatusCache assetStatusCache,
            StatusBar statusBar,
            EditorWindow parentWindow)
        {
            mWkInfo = wkInfo;
            mRepSpec = repSpec;
            mViewHost = viewHost;
            mIsGluonMode = isGluonMode;
            mWorkspaceWindow = workspaceWindow;
            mViewSwitcher = viewSwitcher;
            mShowChangesetInView = showChangesetInView;
            mShowShelveInView = showShelveInView;
            mHistoryViewLauncher = historyViewLauncher;
            mShowDownloadPlasticExeWindow = showDownloadPlasticExeWindow;
            mWorkspaceOperationsMonitor = workspaceOperationsMonitor;
            mSaveAssets = saveAssets;
            mDeveloperNewIncomingChangesUpdater = developerNewIncomingChangesUpdater;
            mGluonNewIncomingChangesUpdater = gluonNewIncomingChangesUpdater;
            mAssetStatusCache = assetStatusCache;
            mStatusBar = statusBar;
            mParentWindow = parentWindow;
            mGuiMessage = new UnityPlasticGuiMessage();
            mCheckedStateManager = new PendingChangesViewCheckedStateManager();

            mNewChangesInWk = NewChangesInWk.Build(
                wkInfo,
                new BuildWorkspacekIsRelevantNewChange());

            BuildComponents(viewSwitcher, isGluonMode);

            mProgressControls = new ProgressControlsForViews();

            if (mErrorsPanel != null)
            {
                mErrorsSplitterState = PlasticSplitterGUILayout.InitSplitterState(
                    new float[] { 0.75f, 0.25f },
                    new int[] { 100, 100 },
                    new int[] { 100000, 100000 }
                );
            }

            workspaceWindow.RegisterPendingChangesProgressControls(mProgressControls);

            mPendingChangesOperations = new PendingChangesOperations(
                wkInfo,
                workspaceWindow,
                viewSwitcher,
                mergeViewLauncher,
                this,
                mProgressControls,
                workspaceWindow,
                null,
                null,
                null,
                null);

            mCommentText = SessionState.GetString(
                UnityConstants.PENDING_CHANGES_CI_COMMENTS_KEY_NAME,
                string.Empty);

            InitIgnoreRulesAndRefreshView(wkInfo.ClientPath, this);
        }

        internal void AutoRefresh()
        {
            if (mIsAutoRefreshDisabled)
                return;

            if (!PlasticGuiConfig.Get().Configuration.CommitAutoRefresh)
                return;

            if (mNewChangesInWk != null && !mNewChangesInWk.Detected())
                return;

            ((IRefreshableView)this).Refresh();
        }

        internal void ClearIsCommentWarningNeeded()
        {
            mIsEmptyCheckinCommentWarningNeeded = false;
            mIsEmptyShelveCommentWarningNeeded = false;
        }

        internal void UpdateIsCheckinCommentWarningNeeded(string comment)
        {
            mIsEmptyCheckinCommentWarningNeeded =
                string.IsNullOrEmpty(comment) &&
                PlasticGuiConfig.Get().Configuration.ShowEmptyCommentWarning;

            mNeedsToShowEmptyCheckinCommentDialog = mIsEmptyCheckinCommentWarningNeeded;
        }

        internal void UpdateIsShelveCommentWarningNeeded(string comment)
        {
            mIsEmptyShelveCommentWarningNeeded =
                string.IsNullOrEmpty(comment) &&
                PlasticGuiConfig.Get().Configuration.ShowEmptyShelveCommentWarning;

            mNeedsToShowEmptyShelveCommentDialog = mIsEmptyShelveCommentWarningNeeded;
        }

        internal void OnEnable()
        {
            mIsEnabled = true;

            mSearchField.downOrUpArrowKeyPressed +=
                SearchField_OnDownOrUpArrowKeyPressed;
        }

        internal void OnDisable()
        {
            mIsEnabled = false;

            mSearchField.downOrUpArrowKeyPressed -=
                SearchField_OnDownOrUpArrowKeyPressed;

            SessionState.SetString(
                UnityConstants.PENDING_CHANGES_CI_COMMENTS_KEY_NAME,
                mCommentText);

            TreeViewSessionState.Save(
                mPendingChangesTreeView,
                UnityConstants.PENDING_CHANGES_UNCHECKED_ITEMS_KEY_NAME);

            TreeHeaderSettings.Save(
                mPendingChangesTreeView.multiColumnHeader.state,
                UnityConstants.PENDING_CHANGES_TABLE_SETTINGS_NAME);

            if (mErrorsPanel != null)
                mErrorsPanel.OnDisable();
        }

        internal void Update()
        {
            mProgressControls.UpdateProgress(mParentWindow);

            // Display the empty comment dialog here, otherwise it causes errors in the OnGUI method
            if (mNeedsToShowEmptyCheckinCommentDialog)
            {
                mNeedsToShowEmptyCheckinCommentDialog = false;

                mHasPendingCheckinFromPreviousUpdate =
                    EmptyCommentDialog.ShouldContinueWithCheckin(mParentWindow, mWkInfo);

                mIsEmptyCheckinCommentWarningNeeded = !mHasPendingCheckinFromPreviousUpdate;
            }

            if (mNeedsToShowEmptyShelveCommentDialog)
            {
                mNeedsToShowEmptyShelveCommentDialog = false;

                mHasPendingShelveFromPreviousUpdate =
                    EmptyCommentDialog.ShouldContinueWithShelve(mParentWindow, mWkInfo);

                mIsEmptyShelveCommentWarningNeeded = !mHasPendingShelveFromPreviousUpdate;
            }
        }

        internal void OnGUI(
            ResolvedUser currentUser,
            Action repaintAction)
        {
            if (mErrorsPanel != null && mErrorsPanel.IsVisible)
                PlasticSplitterGUILayout.BeginVerticalSplit(mErrorsSplitterState);

            DoContentArea();

            if (mErrorsPanel != null && mErrorsPanel.IsVisible)
            {
                mErrorsPanel.OnGUI();
                PlasticSplitterGUILayout.EndVerticalSplit();
            }

            DoSeparator();

            DoCommentsSection(currentUser, repaintAction);

            if (mProgressControls.HasNotification())
                DrawProgressForViews.ForNotificationArea(mProgressControls.ProgressData);

            ExecuteAfterOnGUIAction();
            }

        void DoContentArea()
        {
            EditorGUILayout.BeginVertical();

            if (!string.IsNullOrEmpty(mGluonWarningMessage))
                DoWarningMessage(mGluonWarningMessage);

            DoActionsToolbar(mProgressControls);

            DoChangesArea(
                mPendingChangesTreeView,
                mEmptyStateData,
                mProgressControls.IsOperationRunning(),
                mDrawOperationSuccess,
                mParentWindow.Repaint);

            if (HasPendingMergeLinks() && !mHasPendingMergeLinksFromRevert)
                DoMergeLinksArea(mMergeLinksListView, mParentWindow.position.width);

            EditorGUILayout.EndVertical();
        }

        internal void DrawSearchFieldForTab()
        {
            DrawSearchField.For(
                mSearchField,
                mPendingChangesTreeView,
                UnityConstants.SEARCH_FIELD_WIDTH);
        }

        internal void UpdateComment(string comment)
        {
            mCommentText = comment;
            mForceToShowComment = false;
        }

        void IPendingChangesView.SetDefaultComment(string defaultComment)
        {
        }

        void IPendingChangesView.ClearComments()
        {
            ClearComments();
        }

        void IRefreshableView.Refresh()
        {
            if (mProgressControls.IsOperationRunning())
                return;

            if (!mAreIgnoreRulesInitialized)
                return;

            if (mDeveloperNewIncomingChangesUpdater != null)
                mDeveloperNewIncomingChangesUpdater.Update(DateTime.Now);

            if (mGluonNewIncomingChangesUpdater != null)
                mGluonNewIncomingChangesUpdater.Update(DateTime.Now);

            FillPendingChanges(mNewChangesInWk);
        }

        void PendingChangesOptionsFoldout.IAutoRefreshView.DisableAutoRefresh()
        {
            mIsAutoRefreshDisabled = true;
        }

        void PendingChangesOptionsFoldout.IAutoRefreshView.EnableAutoRefresh()
        {
            mIsAutoRefreshDisabled = false;
        }

        void PendingChangesOptionsFoldout.IAutoRefreshView.ForceRefresh()
        {
            ((IRefreshableView)this).Refresh();
        }

        void IPendingChangesView.ClearChangesToCheck(List<string> changes)
        {
            mCheckedStateManager.ClearChangesToCheck(changes);

            mParentWindow.Repaint();
        }

        void IPendingChangesView.CleanCheckedElements(List<ChangeInfo> checkedChanges)
        {
            mCheckedStateManager.Clean(checkedChanges);

            mParentWindow.Repaint();
        }

        void IPendingChangesView.CheckChanges(List<string> changesToCheck)
        {
            mCheckedStateManager.SetChangesToCheck(changesToCheck);

            mParentWindow.Repaint();
        }

        bool IPendingChangesView.IncludeDependencies(
            IList<ChangeDependencies> changesDependencies,
            string operation)
        {
            return DependenciesDialog.IncludeDependencies(
                mWkInfo, changesDependencies, operation, mParentWindow);
        }

        SearchMatchesData IPendingChangesView.AskForMatches(string changePath)
        {
            throw new NotImplementedException();
        }

        void IPendingChangesView.CleanLinkedTasks()
        {
        }

        void CheckinUIOperation.ICheckinView.CollapseWarningMessagePanel()
        {
            mGluonWarningMessage = string.Empty;

            mParentWindow.Repaint();
        }

        void CheckinUIOperation.ICheckinView.ExpandWarningMessagePanel(string text)
        {
            mGluonWarningMessage = text;

            mParentWindow.Repaint();
        }

        void CheckinUIOperation.ICheckinView.ClearComments()
        {
            ClearComments();
        }

        void ShelveOperations.ICheckinView.OnShelvesetApplied(List<ErrorMessage> errorMessages)
        {
            mViewSwitcher.ShowPendingChanges();
            mErrorsPanel.UpdateErrorsList(errorMessages);
        }

        bool PendingChangesViewPendingChangeMenu.IMetaMenuOperations.SelectionHasMeta()
        {
            return mPendingChangesTreeView.SelectionHasMeta();
        }

        void PendingChangesViewPendingChangeMenu.IMetaMenuOperations.DiffMeta()
        {
            ChangeInfo selectedChange = PendingChangesSelection
                .GetSelectedChange(mPendingChangesTreeView);
            ChangeInfo selectedChangeMeta = mPendingChangesTreeView.GetMetaChange(
                selectedChange);

            ChangeInfo changedForMoved = mPendingChangesTreeView.GetChangedForMoved(selectedChange);
            ChangeInfo changedForMovedMeta = (changedForMoved == null) ?
                null : mPendingChangesTreeView.GetMetaChange(changedForMoved);

            DiffOperation.DiffWorkspaceContent(
                mWkInfo,
                selectedChangeMeta,
                changedForMovedMeta,
                mProgressControls,
                PlasticExeLauncher.BuildForDiffWorkspaceContent(mWkInfo, mIsGluonMode, mShowDownloadPlasticExeWindow),
                null);
        }

        void PendingChangesViewPendingChangeMenu.IMetaMenuOperations.HistoryMeta()
        {
            ChangeInfo selectedChange = PendingChangesSelection
                .GetSelectedChange(mPendingChangesTreeView);
            ChangeInfo selectedChangeMeta = mPendingChangesTreeView.GetMetaChange(
                selectedChange);

            mHistoryViewLauncher.ShowHistoryView(
                selectedChangeMeta.RepositorySpec,
                selectedChangeMeta.RevInfo.ItemId,
                selectedChangeMeta.Path,
                selectedChangeMeta.IsDirectory);
        }

        void PendingChangesViewPendingChangeMenu.IMetaMenuOperations.OpenMeta()
        {
            List<string> selectedPaths = PendingChangesSelection
                .GetSelectedMetaPaths(mPendingChangesTreeView);

            FileSystemOperation.Open(selectedPaths);
        }

        void PendingChangesViewPendingChangeMenu.IMetaMenuOperations.OpenMetaWith()
        {
            List<string> selectedPaths = PendingChangesSelection
                .GetSelectedMetaPaths(mPendingChangesTreeView);

            OpenOperation.OpenWith(
                FileSystemOperation.GetExePath(),
                selectedPaths);
        }

        void PendingChangesViewPendingChangeMenu.IMetaMenuOperations.OpenMetaInExplorer()
        {
            List<string> selectedPaths = PendingChangesSelection
                .GetSelectedMetaPaths(mPendingChangesTreeView);

            if (selectedPaths.Count < 1)
                return;

            FileSystemOperation.OpenInExplorer(selectedPaths[0]);
        }

        SelectedChangesGroupInfo IPendingChangesMenuOperations.GetSelectedChangesGroupInfo()
        {
            return PendingChangesSelection.GetSelectedChangesGroupInfo(
                mWkInfo.ClientPath, mPendingChangesTreeView);
        }

        void IPendingChangesMenuOperations.Diff()
        {
            ChangeInfo selectedChange = PendingChangesSelection
                .GetSelectedChange(mPendingChangesTreeView);

            DiffOperation.DiffWorkspaceContent(
                mWkInfo,
                selectedChange,
                mPendingChangesTreeView.GetChangedForMoved(selectedChange),
                null,
                PlasticExeLauncher.BuildForDiffWorkspaceContent(mWkInfo, mIsGluonMode, mShowDownloadPlasticExeWindow),
                null);
        }

        void IPendingChangesMenuOperations.UndoChanges()
        {
            List<ChangeInfo> changesToUndo = PendingChangesSelection
                .GetSelectedChanges(mPendingChangesTreeView);

            List<ChangeInfo> dependenciesCandidates =
                mPendingChangesTreeView.GetDependenciesCandidates(changesToUndo, true);

            UndoChangesForMode(mIsGluonMode, false, changesToUndo, dependenciesCandidates);
        }

        void IPendingChangesMenuOperations.SearchMatches()
        {
            ChangeInfo selectedChange = PendingChangesSelection
                .GetSelectedChange(mPendingChangesTreeView);

            if (selectedChange == null)
                return;

            SearchMatchesOperation operation = new SearchMatchesOperation(
                mWkInfo, mWorkspaceWindow, this,
                mProgressControls, mDeveloperNewIncomingChangesUpdater, null);

            operation.SearchMatches(
                selectedChange,
                PendingChangesSelection.GetAllChanges(mPendingChangesTreeView),
                null);
        }

        void IPendingChangesMenuOperations.ApplyLocalChanges()
        {
            List<ChangeInfo> selectedChanges = PendingChangesSelection
                .GetSelectedChanges(mPendingChangesTreeView);

            if (selectedChanges.Count == 0)
                return;

            ApplyLocalChangesOperation operation = new ApplyLocalChangesOperation(
                mWkInfo, mWorkspaceWindow, this,
                mProgressControls, mDeveloperNewIncomingChangesUpdater, null);

            operation.ApplyLocalChanges(
                selectedChanges,
                PendingChangesSelection.GetAllChanges(mPendingChangesTreeView),
                null);
        }

        void IPendingChangesMenuOperations.CopyFilePath(bool relativePath)
        {
            EditorGUIUtility.systemCopyBuffer = GetFilePathList.FromSelectedPaths(
                PendingChangesSelection.GetSelectedPathsWithoutMeta(mPendingChangesTreeView),
                relativePath,
                mWkInfo.ClientPath);
        }

        void IPendingChangesMenuOperations.Delete()
        {
            List<string> privateDirectoriesToDelete;
            List<string> privateFilesToDelete;

            if (!mPendingChangesTreeView.GetSelectedPathsToDelete(
                    out privateDirectoriesToDelete,
                    out privateFilesToDelete))
                return;

            DeleteOperation.Delete(
                mProgressControls,
                privateDirectoriesToDelete,
                privateFilesToDelete,
                mDeveloperNewIncomingChangesUpdater,
                null,
                RefreshAsset.UnityAssetDatabase,
                () => ((IWorkspaceWindow)mWorkspaceWindow).RefreshView(ViewType.ItemsView),
                () => ((IWorkspaceWindow)mWorkspaceWindow).RefreshView(ViewType.PendingChangesView));
        }

        void IPendingChangesMenuOperations.Annotate()
        {
            throw new NotImplementedException();
        }

        void IPendingChangesMenuOperations.History()
        {
            ChangeInfo selectedChange = PendingChangesSelection.
                GetSelectedChange(mPendingChangesTreeView);

            mHistoryViewLauncher.ShowHistoryView(
                selectedChange.RepositorySpec,
                selectedChange.RevInfo.ItemId,
                selectedChange.Path,
                selectedChange.IsDirectory);
        }

        SelectedChangesGroupInfo IChangelistMenuOperations.GetSelectedChangesGroupInfo()
        {
            return PendingChangesSelection.GetSelectedChangesGroupInfo(
                mWkInfo.ClientPath, mPendingChangesTreeView);
        }

        List<ChangeListInfo> IChangelistMenuOperations.GetSelectedChangelistInfos()
        {
            return PendingChangesSelection.GetSelectedChangeListInfos(
                mPendingChangesTreeView);
        }

        void IChangelistMenuOperations.Checkin()
        {
            List<ChangeInfo> changesToCheckin;
            List<ChangeInfo> dependenciesCandidates;

            mPendingChangesTreeView.GetCheckedChanges(
                PendingChangesSelection.GetSelectedChangelistNodes(mPendingChangesTreeView),
                false, out changesToCheckin, out dependenciesCandidates);

            CheckinChangesForMode(
                changesToCheckin,
                dependenciesCandidates,
                mIsGluonMode,
                mKeepItemsLocked);
        }

        void IChangelistMenuOperations.Shelve()
        {
            List<ChangeInfo> changesToShelve;
            List<ChangeInfo> dependenciesCandidates;

            mPendingChangesTreeView.GetCheckedChanges(
                PendingChangesSelection.GetSelectedChangelistNodes(mPendingChangesTreeView),
                false, out changesToShelve, out dependenciesCandidates);

            ShelveChangesForMode(
                changesToShelve,
                dependenciesCandidates,
                mIsGluonMode,
                mKeepItemsLocked);
        }

        void IChangelistMenuOperations.Undo()
        {
            List<ChangeInfo> changesToUndo;
            List<ChangeInfo> dependenciesCandidates;

            mPendingChangesTreeView.GetCheckedChanges(
                PendingChangesSelection.GetSelectedChangelistNodes(mPendingChangesTreeView),
                true, out changesToUndo, out dependenciesCandidates);

            UndoChangesForMode(mIsGluonMode, false, changesToUndo, dependenciesCandidates);
        }

        void IChangelistMenuOperations.CreateNew()
        {
            ChangelistCreationData changelistCreationData =
                CreateChangelistDialog.CreateChangelist(mWkInfo, mParentWindow);

            ChangelistOperations.CreateNew(mWkInfo, this, changelistCreationData);
        }

        void IChangelistMenuOperations.MoveToNewChangelist(List<ChangeInfo> changes)
        {
            ChangelistCreationData changelistCreationData =
                CreateChangelistDialog.CreateChangelist(mWkInfo, mParentWindow);

            if (!changelistCreationData.Result)
                return;

            ChangelistOperations.CreateNew(mWkInfo, this, changelistCreationData);

            ChangelistOperations.MoveToChangelist(
                mWkInfo, this, changes,
                changelistCreationData.ChangelistInfo.Name);
        }

        void IChangelistMenuOperations.Edit()
        {
            ChangeListInfo changelistToEdit = PendingChangesSelection.GetSelectedChangeListInfo(
                mPendingChangesTreeView);

            ChangelistCreationData changelistCreationData = CreateChangelistDialog.EditChangelist(
                mWkInfo,
                changelistToEdit,
                mParentWindow);

            ChangelistOperations.Edit(mWkInfo, this, changelistToEdit, changelistCreationData);
        }

        void IChangelistMenuOperations.Delete()
        {
            ChangelistOperations.Delete(
                mWkInfo,
                this,
                PendingChangesSelection.GetSelectedChangelistNodes(mPendingChangesTreeView));
        }

        void IChangelistMenuOperations.MoveToChangelist(
            List<ChangeInfo> changes,
            string targetChangelist)
        {
            ChangelistOperations.MoveToChangelist(
                mWkInfo,
                this,
                changes,
                targetChangelist);
        }

        void IOpenMenuOperations.Open()
        {
            List<string> selectedPaths = PendingChangesSelection.
                GetSelectedPathsWithoutMeta(mPendingChangesTreeView);

            FileSystemOperation.Open(selectedPaths);
        }

        void IOpenMenuOperations.OpenWith()
        {
            List<string> selectedPaths = PendingChangesSelection.
                GetSelectedPathsWithoutMeta(mPendingChangesTreeView);

            OpenOperation.OpenWith(
                FileSystemOperation.GetExePath(),
                selectedPaths);
        }

        void IOpenMenuOperations.OpenWithCustom(string exePath, string args)
        {
            List<string> selectedPaths = PendingChangesSelection.
                GetSelectedPathsWithoutMeta(mPendingChangesTreeView);

            OpenOperation.OpenWith(exePath, selectedPaths);
        }

        void IOpenMenuOperations.OpenInExplorer()
        {
            List<string> selectedPaths = PendingChangesSelection
                .GetSelectedPathsWithoutMeta(mPendingChangesTreeView);

            if (selectedPaths.Count < 1)
                return;

            FileSystemOperation.OpenInExplorer(selectedPaths[0]);
        }

        void IFilesFilterPatternsMenuOperations.AddFilesFilterPatterns(
            FilterTypes type, FilterActions action, FilterOperationType operation)
        {
            List<string> selectedPaths = PendingChangesSelection.GetSelectedPaths(
                mPendingChangesTreeView);

            string[] rules = FilterRulesGenerator.GenerateRules(
                selectedPaths, mWkInfo.ClientPath, action, operation);

            bool isApplicableToAllWorkspaces = !mIsGluonMode;
            bool isAddOperation = operation == FilterOperationType.Add;

            FilterRulesConfirmationData filterRulesConfirmationData =
                FilterRulesConfirmationDialog.AskForConfirmation(
                    rules, isAddOperation, isApplicableToAllWorkspaces, mParentWindow);

            AddFilesFilterPatternsOperation.Run(
                mWkInfo, mWorkspaceWindow, type, operation, filterRulesConfirmationData);
        }

        void PendingChangesViewPendingChangeMenu.IAdvancedUndoMenuOperations.UndoUnchanged()
        {
            UndoUnchangedChanges(PendingChangesSelection.
                GetSelectedChanges(mPendingChangesTreeView));
        }

        void PendingChangesViewPendingChangeMenu.IAdvancedUndoMenuOperations.UndoCheckoutsKeepingChanges()
        {
            UndoCheckoutChangesKeepingLocalChanges(PendingChangesSelection.
                GetSelectedChanges(mPendingChangesTreeView));
        }

        List<IPlasticTreeNode> PendingChangesViewMenu.IGetSelectedNodes.GetSelectedNodes()
        {
            return mPendingChangesTreeView.GetSelectedNodes();
        }

        void ChangesetsTab.IRevertToChangesetListener.OnSuccessOperation()
        {
            mHasPendingMergeLinksFromRevert = true;
        }

        void SearchField_OnDownOrUpArrowKeyPressed()
        {
            mPendingChangesTreeView.SetFocusAndEnsureSelectedItem();
        }

        void ClearOperationSuccess()
        {
            mDrawOperationSuccess = null;
        }

        void InitIgnoreRulesAndRefreshView(
            string wkPath, IRefreshableView view)
        {
            IThreadWaiter waiter = ThreadWaiter.GetWaiter(10);
            waiter.Execute(
                /*threadOperationDelegate*/ delegate
                {
                    if (IsIgnoreConfigDefined.InWorkspace(wkPath))
                    {
                        AddIgnoreRules.WriteRules(
                            wkPath, UnityConditions.GetMissingIgnoredRuleForUnityDirMonSyncFile(wkPath));
                        return;
                    }

                    if (IsIgnoreConfigDefined.InGlobalConfig(wkPath))
                        return;

                    AddIgnoreRules.WriteRules(
                        wkPath, UnityConditions.GetMissingIgnoredRules(wkPath));
                },
                /*afterOperationDelegate*/ delegate
                {
                    mAreIgnoreRulesInitialized = true;

                    view.Refresh();

                    if (waiter.Exception == null)
                        return;

                    mLog.ErrorFormat(
                        "Error adding ignore rules for Unity: {0}",
                        waiter.Exception);

                    mLog.DebugFormat(
                        "Stack trace: {0}",
                        waiter.Exception.StackTrace);
                });
        }

        void FillPendingChanges(INewChangesInWk newChangesInWk)
        {
            if (mIsRefreshing)
                return;

            mIsRefreshing = true;

            ClearOperationSuccess();

            List<ChangeInfo> changesToSelect =
                PendingChangesSelection.GetChangesToFocus(mPendingChangesTreeView);

            ((IProgressControls)mProgressControls).ShowProgress(PlasticLocalization.
                GetString(PlasticLocalization.Name.LoadingPendingChanges));

            IDictionary<MountPoint, IList<PendingMergeLink>> mergeLinks = null;

            WorkspaceStatusResult status = null;

            IThreadWaiter waiter = ThreadWaiter.GetWaiter();
            waiter.Execute(
                /*threadOperationDelegate*/ delegate
                {
                    FilterManager.Get().Reload(mWkInfo);

                    WorkspaceStatusOptions options = WorkspaceStatusOptions.None;
                    options |= WorkspaceStatusOptions.FindAdded;
                    options |= WorkspaceStatusOptions.FindDeleted;
                    options |= WorkspaceStatusOptions.FindMoved;
                    options |= WorkspaceStatusOptions.SplitModifiedMoved;
                    options |= PendingChangesOptions.GetWorkspaceStatusOptions();

                    if (newChangesInWk != null)
                        newChangesInWk.Detected();

                    status = GetStatus.ForWorkspace(
                        mWkInfo,
                        options,
                        PendingChangesOptions.GetMovedMatchingOptions());

                    mergeLinks = PlasticGui.Plastic.API.GetPendingMergeLinks(mWkInfo);
                },
                /*afterOperationDelegate*/ delegate
                {
                    mPendingMergeLinks = mergeLinks;

                    try
                    {
                        if (waiter.Exception != null)
                        {
                            ExceptionsHandler.DisplayException(waiter.Exception);
                            return;
                        }

                        UpdateChangesTree(status.Changes);

                        RestoreData();

                        UpdateMergeLinksList();

                        PendingChangesSelection.SelectChanges(
                            mPendingChangesTreeView, changesToSelect);

                        ClearAssetStatusCache();
                    }
                    finally
                    {
                        ((IProgressControls)mProgressControls).HideProgress();

                        UpdateNotificationPanel();

                        mIsRefreshing = false;
                    }
                });
        }

        void DoSeparator()
        {
            Rect result = GUILayoutUtility.GetRect(mParentWindow.position.width, 1);
            EditorGUI.DrawRect(result, UnityStyles.Colors.BarBorder);
        }

        void DoCommentsSection(
            ResolvedUser currentUser,
            Action repaintAction)
        {
            EditorGUILayout.BeginVertical(UnityStyles.PendingChangesTab.Comment);
            EditorGUILayout.Space(10);

            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.Space(2);

            EditorGUILayout.BeginVertical();
            GUILayout.FlexibleSpace();
            DrawUserIcon.ForPendingChangesTab(
                currentUser,
                repaintAction);
            GUILayout.FlexibleSpace();
            EditorGUILayout.EndVertical();

            float buttonsWidth = 306f;
            float areaWidth = Mathf.Clamp(mParentWindow.position.width, 300f, 820f);
            float width = areaWidth - buttonsWidth;

            DrawCommentTextArea.For(
               this,
               width,
               mProgressControls.IsOperationRunning());

            EditorGUILayout.Space(2);

            // To center the action buttons vertically
            EditorGUILayout.BeginVertical();
            GUILayout.FlexibleSpace();
            DoOperationsToolbar(
              mIsGluonMode,
              mShelveDropdownMenu,
              mUndoDropdownMenu,
              mProgressControls.IsOperationRunning());
            GUILayout.FlexibleSpace();
            EditorGUILayout.EndVertical();

            GUILayout.FlexibleSpace();

            EditorGUILayout.EndHorizontal();

            EditorGUILayout.Space(10);

            EditorGUILayout.EndVertical();
        }

        void DoOperationsToolbar(
            bool isGluonMode,
            GenericMenu shelveDropdownMenu,
            GenericMenu undoDropdownMenu,
            bool isOperationRunning)
        {
            string checkinButtonText = PlasticLocalization.GetString(
                PlasticLocalization.Name.Checkin);
            string shelveButtonText = PlasticLocalization.GetString(
                PlasticLocalization.Name.Shelve);
            string undoButtonText = PlasticLocalization.GetString(
                PlasticLocalization.Name.UndoChanges);

            mOperationButtonWidth = InitializeOperationButtonWidth(
                DrawActionButton.ButtonStyle,
                DrawActionButtonWithMenu.ButtonStyle,
                mOperationButtonWidth,
                checkinButtonText, shelveButtonText, undoButtonText);

            EditorGUILayout.BeginHorizontal();

            using (new GuiEnabled(!isOperationRunning))
            {
                if (mHasPendingCheckinFromPreviousUpdate)
                {
                    mHasPendingCheckinFromPreviousUpdate = false;
                    CheckinForMode(isGluonMode, mKeepItemsLocked);
                }

                if (mHasPendingShelveFromPreviousUpdate)
                {
                    mHasPendingShelveFromPreviousUpdate = false;
                    ShelveForMode(isGluonMode, mKeepItemsLocked);
                }

                if (DrawActionButton.ForCommentSection(
                        checkinButtonText, mOperationButtonWidth))
                {
                    CheckinAction(isGluonMode);
                }

                GUILayout.Space(2);

                DrawActionButtonWithMenu.For(
                    shelveButtonText, string.Empty, mOperationButtonWidth,
                    () => ShelveAction(isGluonMode),
                    shelveDropdownMenu);

                GUILayout.Space(2);

                DoUndoButton(isGluonMode, undoButtonText, mOperationButtonWidth, undoDropdownMenu);

                if (isGluonMode)
                {
                    mKeepItemsLocked = EditorGUILayout.ToggleLeft(
                        PlasticLocalization.GetString(PlasticLocalization.Name.KeepLocked),
                        mKeepItemsLocked,
                        GUILayout.Width(UnityConstants.EXTRA_LARGE_BUTTON_WIDTH));
                }
            }

            EditorGUILayout.EndHorizontal();
        }

        void DoUndoButton(
            bool isGluonMode,
            string undoButtonText,
            float buttonWidth,
            GenericMenu undoDropdownMenu)
        {
            if (isGluonMode)
            {
                if (DrawActionButton.ForCommentSection(undoButtonText, buttonWidth))
                    UndoChangesAction(true);

                return;
            }

            DrawActionButtonWithMenu.For(
                undoButtonText, string.Empty, buttonWidth,
                () => UndoChangesAction(false),
                undoDropdownMenu);
        }

        void CheckinAction(bool isGluonMode)
        {
            UpdateIsCheckinCommentWarningNeeded(mCommentText);

            if (mIsEmptyCheckinCommentWarningNeeded)
                return;

            CheckinForMode(isGluonMode, mKeepItemsLocked);
        }

        void ShelveAction(bool isGluonMode)
        {
            UpdateIsShelveCommentWarningNeeded(mCommentText);

            if (mIsEmptyShelveCommentWarningNeeded)
                return;

            ShelveForMode(isGluonMode, mKeepItemsLocked);
        }

        void UndoChangesAction(bool isGluonMode)
        {
            TrackFeatureUseEvent.For(
                mRepSpec,
                TrackFeatureUseEvent.Features.UndoTextButton);

            UndoForMode(isGluonMode, false);
        }

        void UpdateChangesTree(List<ChangeInfo> changes)
        {
            mPendingChangesTreeView.BuildModel(changes, mCheckedStateManager);

            mPendingChangesTreeView.Refilter();

            mPendingChangesTreeView.Sort();

            mPendingChangesTreeView.Reload();
        }

        static void DoWarningMessage(string message)
        {
            GUILayout.Label(message, UnityStyles.WarningMessage);
        }

        void UpdateMergeLinksList()
        {
            mMergeLinksListView.BuildModel(mPendingMergeLinks);

            mMergeLinksListView.Reload();

            if (!HasPendingMergeLinks())
                mHasPendingMergeLinksFromRevert = false;
        }

        void ClearAssetStatusCache()
        {
            if (mAssetStatusCache != null)
                mAssetStatusCache.Clear();

            ProjectWindow.Repaint();
            RepaintInspector.All();
        }

        void UpdateNotificationPanel()
        {
            if (PlasticGui.Plastic.API.IsFsReaderWatchLimitReached(mWkInfo))
            {
                ((IProgressControls)mProgressControls).ShowWarning(PlasticLocalization.
                    GetString(PlasticLocalization.Name.NotifyLinuxWatchLimitWarning));
                return;
            }
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

            EditorGUILayout.EndHorizontal();
        }

        void DoChangesArea(
            PendingChangesTreeView changesTreeView,
            EmptyStateData emptyStateData,
            bool isOperationRunning,
            IDrawOperationSuccess drawOperationSuccess,
            Action repaint)
        {
            using (new EditorGUI.DisabledScope(isOperationRunning))
            {
                Rect rect = GUILayoutUtility.GetRect(0, 100000, 0, 100000);
                changesTreeView.OnGUI(rect);

                if (isOperationRunning)
                    return;

                if (changesTreeView.GetTotalItemCount() == 0)
                {
                    DrawEmptyState(
                        rect,
                        emptyStateData,
                        drawOperationSuccess,
                        repaint);
                    return;
                }

                if (drawOperationSuccess != null)
                {
                    drawOperationSuccess.InStatusBar(mStatusBar);
                    mDrawOperationSuccess = null;
                }
            }
        }

        void ExecuteAfterOnGUIAction()
        {
            if (mProgressControls.IsOperationRunning())
                return;

            if (mAfterOnGUIAction == null)
                return;

            mAfterOnGUIAction();
            mAfterOnGUIAction = null;
        }

        static void DrawEmptyState(
            Rect rect,
            EmptyStateData emptyStateData,
            IDrawOperationSuccess drawOperationSuccess,
            Action repaint)
        {
            if (drawOperationSuccess == null)
            {
                DrawNoPendingChangesEmptyState(rect, emptyStateData, repaint);
                return;
            }

            drawOperationSuccess.InEmptyState(rect);
        }

        static void DrawNoPendingChangesEmptyState(
            Rect rect,
            EmptyStateData emptyStateData,
            Action repaint)
        {
            emptyStateData.Update(
                PlasticLocalization.Name.EmptyPendingChangesMessage.GetString(),
                rect, Event.current.type, repaint);

            DrawTreeViewEmptyState.For(emptyStateData);
        }

        bool HasPendingMergeLinks()
        {
            if (mPendingMergeLinks == null)
                return false;

            return mPendingMergeLinks.Count > 0;
        }

        static void DoMergeLinksArea(
            MergeLinksListView mergeLinksListView, float width)
        {
            GUILayout.Label(
                PlasticLocalization.GetString(
                    PlasticLocalization.Name.MergeLinkDescriptionColumn),
                EditorStyles.boldLabel);

            float desiredTreeHeight = mergeLinksListView.DesiredHeight;

            Rect treeRect = GUILayoutUtility.GetRect(
                0,
                width,
                desiredTreeHeight,
                desiredTreeHeight);

            mergeLinksListView.OnGUI(treeRect);
        }

        void ClearComments()
        {
            mCommentText = string.Empty;
            mForceToShowComment = true;

            SessionState.EraseString(UnityConstants.PENDING_CHANGES_CI_COMMENTS_KEY_NAME);

            mParentWindow.Repaint();
        }

        void RestoreData()
        {
            if (!mRestoreData || !mIsEnabled)
                return;

            TreeViewSessionState.Restore(
                mPendingChangesTreeView,
                UnityConstants.PENDING_CHANGES_UNCHECKED_ITEMS_KEY_NAME);

            mRestoreData = false;
        }

        static float InitializeOperationButtonWidth(
            GUIStyle actionButtonStyle,
            GUIStyle actionButtonWithMenuStyle,
            float operationButtonWidth,
            params string[] texts)
        {
            if (operationButtonWidth != -1)
                return operationButtonWidth;

            float actionButtonWidth = MeasureMaxWidth.
                ForTexts(actionButtonStyle, texts);
            float actionButtonWithMenuWidth = MeasureMaxWidth.
                ForTexts(actionButtonWithMenuStyle, texts);

            return Math.Max(actionButtonWidth, actionButtonWithMenuWidth);
        }

        void BuildComponents(
            IViewSwitcher viewSwitcher,
            bool isGluonMode)
        {
            mShelveDropdownMenu = new GenericMenu();
            mShelveDropdownMenu.AddItem(
                new GUIContent(PlasticLocalization.GetString(
                    PlasticLocalization.Name.ShowShelvesButton)),
                false,
                () => ShowShelvesView(viewSwitcher));

            mUndoDropdownMenu = new GenericMenu();
            mUndoDropdownMenu.AddItem(
                new GUIContent(PlasticLocalization.GetString(
                    PlasticLocalization.Name.UndoUnchangedButton)),
                false, UndoUnchanged);
            mUndoDropdownMenu.AddItem(
                new GUIContent(PlasticLocalization.GetString(
                    PlasticLocalization.Name.UndoCheckoutsKeepingChanges)),
                false, UndoCheckoutsKeepingLocalChanges);

            mSearchField = new SearchField();
            mSearchField.downOrUpArrowKeyPressed += SearchField_OnDownOrUpArrowKeyPressed;

            PendingChangesTreeHeaderState headerState =
                PendingChangesTreeHeaderState.GetDefault(isGluonMode);
            TreeHeaderSettings.Load(headerState,
                UnityConstants.PENDING_CHANGES_TABLE_SETTINGS_NAME,
                (int)PendingChangesTreeColumn.Item, true);

            mPendingChangesTreeView = new PendingChangesTreeView(
                mWkInfo, isGluonMode, headerState,
                PendingChangesTreeHeaderState.GetColumnNames(),
                new PendingChangesViewMenu(
                    mWkInfo, this, this, this, this, this, this, this, isGluonMode),
                mAssetStatusCache);
            mPendingChangesTreeView.Reload();

            mMergeLinksListView = new MergeLinksListView();
            mMergeLinksListView.Reload();

            if (isGluonMode)
                mErrorsPanel = new ErrorsPanel(
                    PlasticLocalization.Name.ChangesCannotBeApplied.GetString(),
                    UnityConstants.PENDING_CHANGES_ERRORS_TABLE_SETTINGS_NAME);
        }

        float mOperationButtonWidth = -1;

        GenericMenu mShelveDropdownMenu;
        GenericMenu mUndoDropdownMenu;
        SearchField mSearchField;
        PendingChangesTreeView mPendingChangesTreeView;
        MergeLinksListView mMergeLinksListView;
        ErrorsPanel mErrorsPanel;
        object mErrorsSplitterState;

        IDictionary<MountPoint, IList<PendingMergeLink>> mPendingMergeLinks;

        volatile bool mAreIgnoreRulesInitialized = false;

        bool mIsRefreshing;
        bool mIsAutoRefreshDisabled;
        bool mIsEmptyCheckinCommentWarningNeeded = false;
        bool mNeedsToShowEmptyCheckinCommentDialog = false;
        bool mHasPendingCheckinFromPreviousUpdate = false;
        bool mIsEmptyShelveCommentWarningNeeded = false;
        bool mNeedsToShowEmptyShelveCommentDialog = false;
        bool mHasPendingShelveFromPreviousUpdate = false;
        bool mHasPendingMergeLinksFromRevert = false;
        bool mKeepItemsLocked;
        bool mForceToShowComment;
        IDrawOperationSuccess mDrawOperationSuccess;
        bool mRestoreData = true;
        bool mIsEnabled = true;
        string mCommentText;
        string mGluonWarningMessage;
        Action mAfterOnGUIAction;

        readonly EmptyStateData mEmptyStateData = new EmptyStateData();
        readonly INewChangesInWk mNewChangesInWk;
        readonly ProgressControlsForViews mProgressControls;
        readonly EditorWindow mParentWindow;
        readonly StatusBar mStatusBar;
        readonly IAssetStatusCache mAssetStatusCache;
        readonly PendingChangesOperations mPendingChangesOperations;
        readonly PendingChangesViewCheckedStateManager mCheckedStateManager;
        readonly GuiMessage.IGuiMessage mGuiMessage;
        readonly NewIncomingChangesUpdater mDeveloperNewIncomingChangesUpdater;
        readonly GluonNewIncomingChangesUpdater mGluonNewIncomingChangesUpdater;
        readonly WorkspaceOperationsMonitor mWorkspaceOperationsMonitor;
        readonly ISaveAssets mSaveAssets;
        readonly LaunchTool.IShowDownloadPlasticExeWindow mShowDownloadPlasticExeWindow;
        readonly IHistoryViewLauncher mHistoryViewLauncher;
        readonly WorkspaceWindow mWorkspaceWindow;
        readonly IViewSwitcher mViewSwitcher;
        readonly IShowChangesetInView mShowChangesetInView;
        readonly IShowShelveInView mShowShelveInView;
        readonly ViewHost mViewHost;
        readonly WorkspaceInfo mWkInfo;
        readonly bool mIsGluonMode;
        readonly RepositorySpec mRepSpec;

        static readonly ILog mLog = PlasticApp.GetLogger("PendingChangesTab");
    }
}
