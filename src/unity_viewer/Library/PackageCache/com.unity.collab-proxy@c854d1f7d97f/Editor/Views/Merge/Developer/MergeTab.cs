using System;
using System.Collections.Generic;

using UnityEditor;
using UnityEngine;

using Codice.Client.BaseCommands;
using Codice.Client.BaseCommands.Merge;
using Codice.Client.Commands;
using Codice.Client.Common;
using Codice.Client.Common.FsNodeReaders;
using Codice.Client.Common.Threading;
using Codice.CM.Common;
using Codice.CM.Common.Merge;
using Codice.CM.Common.Mount;
using PlasticGui;
using PlasticGui.WorkspaceWindow;
using PlasticGui.WorkspaceWindow.BranchExplorer;
using PlasticGui.WorkspaceWindow.Topbar;
using PlasticGui.WorkspaceWindow.Diff;
using PlasticGui.WorkspaceWindow.Merge;
using Unity.PlasticSCM.Editor.AssetUtils;
using Unity.PlasticSCM.Editor.Settings;
using Unity.PlasticSCM.Editor.Tool;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.Progress;
using Unity.PlasticSCM.Editor.UI.StatusBar;
using Unity.PlasticSCM.Editor.UI.Tree;
using Unity.PlasticSCM.Editor.Views.Merge.Developer.DirectoryConflicts;
using UnityEditor.IMGUI.Controls;
using ObjectInfo = Codice.CM.Common.ObjectInfo;

namespace Unity.PlasticSCM.Editor.Views.Merge.Developer
{
    internal class MergeTab :
        IIncomingChangesTab,
        IRefreshableView,
        IMergeView,
        IMergeViewMenuOperations,
        MergeViewFileConflictMenu.IMetaMenuOperations
    {

        internal MergeTreeView Table { get { return mMergeTreeView; } }
        internal ConflictResolutionState ConflictResolutionState { get { return mConflictResolutionState; } }
        internal int DirectoryConflictCount { get { return mDirectoryConflictCount; } }
        internal bool IsProcessingMerge { get { return mMergeViewLogic.IsProcessingMerge; } }
        internal GUIContent ValidationLabel { get { return mValidationLabel; } }

        internal static MergeTab Build(
            WorkspaceInfo wkInfo,
            RepositorySpec repSpec,
            IWorkspaceWindow workspaceWindow,
            IViewSwitcher switcher,
            LaunchTool.IShowDownloadPlasticExeWindow showDownloadPlasticExeWindow,
            IHistoryViewLauncher historyViewLauncher,
            NewIncomingChangesUpdater newIncomingChangesUpdater,
            StatusBar statusBar,
            EditorWindow parentWindow,
            ObjectInfo objectInfo,
            ObjectInfo ancestorChangesetInfo,
            EnumMergeType mergeType,
            ShowIncomingChangesFrom from,
            PlasticNotifier plasticNotifier,
            MergeViewLogic.IMergeController mergeController,
            MergeViewLogic.IGetWorkingBranch getWorkingBranch,
            CheckShelvedChanges.IUpdateShelvedChangesNotification updateShelvedChangesNotification,
            IShelvedChangesUpdater shelvedChangesUpdater,
            IRefreshView refreshView,
            bool isIncomingMerge,
            bool isMergeFinished,
            bool showDiscardChangesButton)
        {
            MergeTab mergeTab = new MergeTab(
                wkInfo,
                repSpec,
                workspaceWindow,
                switcher,
                showDownloadPlasticExeWindow,
                historyViewLauncher,
                newIncomingChangesUpdater,
                statusBar,
                parentWindow,
                objectInfo,
                ancestorChangesetInfo,
                mergeType,
                from,
                plasticNotifier,
                mergeController,
                getWorkingBranch,
                updateShelvedChangesNotification,
                shelvedChangesUpdater,
                refreshView,
                isIncomingMerge,
                isMergeFinished,
                showDiscardChangesButton);

            if (!isMergeFinished)
                ((IRefreshableView)mergeTab).Refresh();

            return mergeTab;
        }

        internal static MergeTab BuildFromCalculatedMerge(
            WorkspaceInfo wkInfo,
            RepositorySpec repSpec,
            IWorkspaceWindow workspaceWindow,
            IViewSwitcher switcher,
            LaunchTool.IShowDownloadPlasticExeWindow showDownloadPlasticExeWindow,
            IHistoryViewLauncher historyViewLauncher,
            NewIncomingChangesUpdater newIncomingChangesUpdater,
            StatusBar statusBar,
            EditorWindow parentWindow,
            ObjectInfo objectInfo,
            ObjectInfo ancestorChangesetInfo,
            EnumMergeType mergeType,
            ShowIncomingChangesFrom from,
            PlasticNotifier plasticNotifier,
            MergeViewLogic.IMergeController mergeController,
            MergeViewLogic.IGetWorkingBranch getWorkingBranch,
            CheckShelvedChanges.IUpdateShelvedChangesNotification updateShelvedChangesNotification,
            IShelvedChangesUpdater shelvedChangesUpdater,
            IRefreshView refreshView,
            bool isIncomingMerge,
            bool isMergeFinished,
            CalculatedMergeResult calculatedMergeResult,
            bool showDiscardChangesButton)
        {
            MergeTab mergeTab = new MergeTab(
                wkInfo,
                repSpec,
                workspaceWindow,
                switcher,
                showDownloadPlasticExeWindow,
                historyViewLauncher,
                newIncomingChangesUpdater,
                statusBar,
                parentWindow,
                objectInfo,
                ancestorChangesetInfo,
                mergeType,
                from,
                plasticNotifier,
                mergeController,
                getWorkingBranch,
                updateShelvedChangesNotification,
                shelvedChangesUpdater,
                refreshView,
                isIncomingMerge,
                isMergeFinished,
                showDiscardChangesButton);

            mergeTab.mMergeViewLogic.CalculateMergeFromMergeResult(calculatedMergeResult);

            mergeTab.ProcessAllMerges();

            return mergeTab;
        }

        MergeTab(
            WorkspaceInfo wkInfo,
            RepositorySpec repSpec,
            IWorkspaceWindow workspaceWindow,
            IViewSwitcher switcher,
            LaunchTool.IShowDownloadPlasticExeWindow showDownloadPlasticExeWindow,
            IHistoryViewLauncher historyViewLauncher,
            NewIncomingChangesUpdater newIncomingChangesUpdater,
            StatusBar statusBar,
            EditorWindow parentWindow,
            ObjectInfo objectInfo,
            ObjectInfo ancestorChangesetInfo,
            EnumMergeType mergeType,
            ShowIncomingChangesFrom from,
            PlasticNotifier plasticNotifier,
            MergeViewLogic.IMergeController mergeController,
            MergeViewLogic.IGetWorkingBranch getWorkingBranch,
            CheckShelvedChanges.IUpdateShelvedChangesNotification updateShelvedChangesNotification,
            IShelvedChangesUpdater shelvedChangesUpdater,
            IRefreshView refreshView,
            bool isIncomingMerge,
            bool isMergeFinished,
            bool showDiscardChangesButton)
        {
            mWkInfo = wkInfo;
            mWorkspaceWindow = workspaceWindow;
            mSwitcher = switcher;
            mShowDownloadPlasticExeWindow = showDownloadPlasticExeWindow;
            mHistoryViewLauncher = historyViewLauncher;
            mNewIncomingChangesUpdater = newIncomingChangesUpdater;
            mParentWindow = parentWindow;
            mStatusBar = statusBar;
            mObjectInfo = objectInfo;
            mAncestorObjectInfo = ancestorChangesetInfo;
            mMergeType = mergeType;
            mFrom = from;
            mGuiMessage = new UnityPlasticGuiMessage();
            mMergeController = mergeController;
            mUpdateShelvedChangesNotification = updateShelvedChangesNotification;
            mShelvedChangesUpdater = shelvedChangesUpdater;
            mRefreshView = refreshView;
            mIsIncomingMerge = isIncomingMerge;
            mShowDiscardChangesButton = showDiscardChangesButton;
            mIsMergeFinished = isMergeFinished;

            mIsMergeTo = MergeTypeClassifier.IsMergeTo(mergeType);
            mRepSpec = PlasticGui.Plastic.API.GetRepositorySpec(mWkInfo);
            mTitleText = MergeViewTitle.Get(objectInfo, ancestorChangesetInfo, mergeType);

            BuildComponents(
                mWkInfo,
                mIsIncomingMerge,
                mIsMergeTo,
                mMergeController.IsShelvesetMerge());

            mMergeDialogParameters = PlasticGui.WorkspaceWindow.Merge.MergeSourceBuilder.
                BuildMergeDialogParameters(mergeType, mRepSpec);

            mMergeController.SetMergeDialogParameters(mMergeDialogParameters);

            mProgressControls = new ProgressControlsForViews();

            mCooldownClearUpdateSuccessAction = new CooldownWindowDelayer(
                DelayedClearUpdateSuccess,
                UnityConstants.NOTIFICATION_CLEAR_INTERVAL);

            mMergeViewLogic = new MergeViewLogic(
                mWkInfo,
                repSpec,
                mergeType,
                mIsIncomingMerge,
                mMergeController,
                getWorkingBranch,
                plasticNotifier,
                from,
                null,
                mNewIncomingChangesUpdater,
                shelvedChangesUpdater,
                null,
                this,
                NewChangesInWk.Build(mWkInfo, new BuildWorkspacekIsRelevantNewChange()),
                mProgressControls,
                null);
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
                mMergeTreeView.multiColumnHeader.state,
                mIsIncomingMerge
                    ? UnityConstants.DEVELOPER_INCOMING_CHANGES_TABLE_SETTINGS_NAME
                    : UnityConstants.DEVELOPER_MERGE_TABLE_SETTINGS_NAME);
        }

        internal SerializableMergeTabState GetSerializableState()
        {
            return new SerializableMergeTabState(
                mRepSpec,
                mObjectInfo,
                mAncestorObjectInfo,
                mMergeType,
                mFrom,
                mIsIncomingMerge,
                mIsMergeFinished);
        }

        internal void Update()
        {
            mProgressControls.UpdateProgress(mParentWindow);
        }

        internal void OnGUI()
        {
            if (Event.current.type == EventType.Layout)
            {
                mHasPendingDirectoryConflicts = mMergeChangesTree != null &&
                    MergeChangesTreeParser.GetUnsolvedDirectoryConflictsCount(mMergeChangesTree) > 0;
                mIsOperationRunning = mProgressControls.IsOperationRunning();
            }

            DoTitle(mTitleText, mMergeDialogParameters, mMergeViewLogic);

            DoConflictsTree(
                mMergeTreeView,
                mEmptyStateData,
                mCooldownClearUpdateSuccessAction,
                mIsOperationRunning,
                mIsUpdateSuccessful,
                mIsIncomingMerge,
                mParentWindow.Repaint);

            List<MergeChangeInfo> selectedMergeChanges =
                mMergeTreeView.GetSelectedMergeChanges();

            if (MergeSelection.GetSelectedGroupInfo(
                    mMergeTreeView, mIsIncomingMerge).IsDirectoryConflictsSelection &&
                !Mouse.IsRightMouseButtonPressed(Event.current))
            {
                DoDirectoryConflictResolutionPanel(
                    selectedMergeChanges,
                    new Action<MergeChangeInfo>(ResolveDirectoryConflict),
                    mConflictResolutionStates,
                    mValidationLabel,
                    mMergeDialogParameters.CherryPicking,
                    mIsDirectoryResolutionPanelEnabled,
                    ref mConflictResolutionState);
            }

            DrawActionToolbar.Begin(mParentWindow);

            if (!mIsOperationRunning)
            {
                DoActionToolbarMessage(
                    mIsMessageLabelVisible,
                    mMessageLabelText,
                    mIsMergeFinished,
                    mIsErrorMessageLabelVisible,
                    mErrorMessageLabelText,
                    mDirectoryConflictCount,
                    mFileConflictCount,
                    mChangesSummary,
                    mSwitcher);

                if (mIsProcessMergesButtonVisible && !mIsMergeFinished)
                {
                    DoProcessMergesButton(
                        mIsProcessMergesButtonEnabled && !mHasPendingDirectoryConflicts,
                        mProcessMergesButtonText,
                        mSwitcher,
                        mShowDownloadPlasticExeWindow,
                        mWorkspaceWindow,
                        mGuiMessage,
                        mMergeViewLogic,
                        mMergeDialogParameters.Options.Contributor,
                        mWkInfo,
                        AfterProcessMerges,
                        MergeSuccessfullyFinished);

                    if (mShowDiscardChangesButton)
                    {
                        DoDiscardChangesButton(
                            mWkInfo,
                            (ChangesetInfo)mObjectInfo,
                            mUpdateShelvedChangesNotification,
                            mShelvedChangesUpdater,
                            mRefreshView,
                            mSwitcher);
                    }
                }

                if (mIsCancelMergesButtonVisible)
                {
                    mIsCancelMergesButtonEnabled = DoCancelMergesButton(
                        mIsCancelMergesButtonEnabled,
                        mMergeViewLogic);
                }

                if (mHasPendingDirectoryConflicts)
                {
                    GUILayout.Space(5);
                    DoWarningMessage();
                }
            }
            else
            {
                DrawProgressForViews.ForIndeterminateProgress(
                    mProgressControls.ProgressData);
            }

            DrawActionToolbar.End();

            if (mProgressControls.HasNotification())
            {
                DrawProgressForViews.ForNotificationArea(
                    mProgressControls.ProgressData);
            }
        }

        internal void DrawSearchFieldForTab()
        {
            DrawSearchField.For(
                mSearchField,
                mMergeTreeView,
                UnityConstants.SEARCH_FIELD_WIDTH);
        }

        internal void AutoRefresh()
        {
            mMergeViewLogic.AutoRefresh();
        }

        internal void ProcessMergeForTesting()
        {
            ProcessMerges(
                new List<string>(),
                null,
                mMergeDialogParameters.Options.Contributor);
        }

        internal void ResolveDirectoryConflict(MergeChangeInfo conflict)
        {
            ConflictResolutionState state;

            if (!mConflictResolutionStates.TryGetValue(conflict.DirectoryConflict, out state))
                return;

            List<DirectoryConflictResolutionData> conflictResolutions =
                new List<DirectoryConflictResolutionData>();

            AddConflictResolution(
                conflict,
                state.ResolveAction,
                state.RenameValue,
                conflictResolutions);

            MergeChangeInfo metaConflict =
                mMergeTreeView.GetMetaChange(conflict);

            if (metaConflict != null)
            {
                AddConflictResolution(
                    metaConflict,
                    state.ResolveAction,
                    MetaPath.GetMetaPath(state.RenameValue),
                    conflictResolutions);
            }

            if (state.IsApplyActionsForNextConflictsChecked)
            {
                foreach (MergeChangeInfo otherConflict in mMergeTreeView.GetSelectedMergeChanges())
                {
                    AddConflictResolution(
                        otherConflict,
                        state.ResolveAction,
                        state.RenameValue,
                        conflictResolutions);
                }
            }

            mMergeViewLogic.ResolveDirectoryConflicts(conflictResolutions);
        }

        bool IIncomingChangesTab.IsVisible{ get; set; }

        void IIncomingChangesTab.OnEnable()
        {
            OnEnable();
        }

        void IIncomingChangesTab.OnDisable()
        {
            OnDisable();
        }

        void IIncomingChangesTab.Update()
        {
            Update();
        }

        void IIncomingChangesTab.DrawSearchFieldForTab()
        {
            DrawSearchFieldForTab();
        }

        void IIncomingChangesTab.AutoRefresh()
        {
            AutoRefresh();
        }

        void IRefreshableView.Refresh()
        {
            mMergeViewLogic.Refresh();
        }

        void IMergeView.UpdateTitle(string title)
        {
            mTitleText = title;
        }

        void IIncomingChangesTab.OnGUI()
        {
            OnGUI();
        }

        void IMergeView.UpdateData(
            MergeChangesTree mergeChangesTree,
            ExplainMergeData explainMergeData,
            MergeSolvedFileConflicts solvedFileConflicts,
            bool isIncomingMerge,
            bool isMergeTo,
            bool mergeHasFinished)
        {
            HideMessage();

            ShowProcessMergesButton(
                MergeViewTexts.GetProcessMergesButtonText(
                    MergeChangesTreeParser.HasFileConflicts(mergeChangesTree),
                    mIsIncomingMerge,
                    mIsMergeTo));

            mMergeChangesTree = mergeChangesTree;

            mConflictResolutionStates.Clear();

            UpdateFileConflictsTree(
                mergeChangesTree,
                mMergeTreeView);

            UpdateOverview(mergeChangesTree, solvedFileConflicts);
        }

        void IMergeView.UpdateSolvedDirectoryConflicts()
        {
            if (mMergeChangesTree == null)
                return;

            mDirectoryConflictCount = MergeChangesTreeParser.GetUnsolvedDirectoryConflictsCount(
                mMergeChangesTree);
        }

        void IMergeView.UpdateSolvedFileConflicts(
            MergeSolvedFileConflicts solvedFileConflicts)
        {
            mMergeTreeView.UpdateSolvedFileConflicts(
                solvedFileConflicts);
        }

        void IMergeView.ShowMessage(
            string title,
            string message,
            bool isErrorMessage)
        {
            if (isErrorMessage)
            {
                mErrorMessageLabelText = message;
                mIsErrorMessageLabelVisible = true;
                return;
            }

            mMessageLabelText = message;
            mIsMessageLabelVisible = true;
            mIsMergeFinished = MergeViewTexts.IsMergeAlreadyProcessedMessage(message);
        }

        string IMergeView.GetComments(out bool bCancel)
        {
            bCancel = false;
            return string.Empty;
        }

        ObjectInfo IMergeView.GetObjectInfo()
        {
            return mObjectInfo;
        }

        void IMergeView.DisableProcessMergesButton()
        {
            mIsProcessMergesButtonEnabled = false;
        }

        void IMergeView.EnableResolveDirectoryConflictsControls()
        {
            mIsDirectoryResolutionPanelEnabled = true;
        }

        void IMergeView.DisableResolveDirectoryConflictsControls()
        {
            mIsDirectoryResolutionPanelEnabled = false;
        }

        void IMergeView.ShowCancelButton()
        {
            mIsCancelMergesButtonEnabled = true;
            mIsCancelMergesButtonVisible = true;
        }

        void IMergeView.HideCancelButton()
        {
            mIsCancelMergesButtonEnabled = false;
            mIsCancelMergesButtonVisible = false;
        }

        void IMergeViewMenuOperations.MergeContributors()
        {
            ProcessMerges(
                MergeSelection.GetPathsFromSelectedFileConflictsIncludingMeta(mMergeTreeView),
                PlasticExeLauncher.BuildForMergeSelectedFiles(mWkInfo, false, mShowDownloadPlasticExeWindow),
                MergeContributorType.MergeContributors);
        }

        void IMergeViewMenuOperations.MergeKeepingSourceChanges()
        {
            ProcessMerges(
                MergeSelection.GetPathsFromSelectedFileConflictsIncludingMeta(mMergeTreeView),
                null,
                MergeContributorType.KeepSource);
        }

        void IMergeViewMenuOperations.MergeKeepingWorkspaceChanges()
        {
            ProcessMerges(
                MergeSelection.GetPathsFromSelectedFileConflictsIncludingMeta(mMergeTreeView),
                null,
                MergeContributorType.KeepDestination);
        }

        SelectedMergeChangesGroupInfo IMergeViewMenuOperations.GetSelectedMergeChangesGroupInfo()
        {
            return GetSelectedMergeChangesGroupInfo.For(
                mMergeTreeView.GetSelectedMergeChanges(), mIsIncomingMerge);
        }

        void IMergeViewMenuOperations.DiffSourceWithDestination()
        {
            MergeChangeInfo mergeChange = MergeSelection.
                GetSingleSelectedMergeChange(mMergeTreeView);

            if (mergeChange == null)
                return;

            if (mergeChange.DirectoryConflict != null)
            {
                DiffSourceWithDestinationForDirectoryConflict(
                    mShowDownloadPlasticExeWindow,
                    mergeChange,
                    mWkInfo,
                    mIsIncomingMerge);
                return;
            }

            DiffSourceWithDestinationForFileConflict(
                mShowDownloadPlasticExeWindow,
                mergeChange,
                mWkInfo,
                mIsIncomingMerge);
        }

        void IMergeViewMenuOperations.DiffDestinationWithAncestor()
        {
            MergeChangeInfo mergeChange = MergeSelection.
                GetSingleSelectedMergeChange(mMergeTreeView);

            if (mergeChange == null)
                return;

            if (mergeChange.DirectoryConflict != null)
            {
                DiffDestinationWithAncestorForDirectoryConflict(
                    mShowDownloadPlasticExeWindow,
                    mergeChange,
                    mWkInfo,
                    mIsIncomingMerge);
                return;
            }

            DiffDestinationWithAncestorForFileConflict(
                mShowDownloadPlasticExeWindow,
                mergeChange,
                mWkInfo);
        }

        void IMergeViewMenuOperations.DiffSourceWithAncestor()
        {
            MergeChangeInfo mergeChange = MergeSelection.
                GetSingleSelectedMergeChange(mMergeTreeView);

            if (mergeChange == null)
                return;

            if (mergeChange.DirectoryConflict != null)
            {
                DiffSourceWithAncestorForDirectoryConflict(
                    mShowDownloadPlasticExeWindow,
                    mergeChange,
                    mWkInfo,
                    mIsIncomingMerge);
                return;
            }

            DiffSourceWithAncestorForFileConflict(
                mShowDownloadPlasticExeWindow,
                mergeChange,
                mWkInfo);
        }

        void IMergeViewMenuOperations.OpenSrcRevision()
        {
            MergeChangeInfo mergeChange = MergeSelection.
                GetSingleSelectedMergeChange(mMergeTreeView);

            if (mergeChange == null)
                return;

            OpenRevision.ForDifference(mRepSpec, mergeChange.DirectoryConflict.SrcDiff);
        }

        void IMergeViewMenuOperations.OpenDstRevision()
        {
            MergeChangeInfo mergeChange = MergeSelection.
                GetSingleSelectedMergeChange(mMergeTreeView);

            if (mergeChange == null)
                return;

            OpenRevision.ForDifference(mRepSpec, mergeChange.DirectoryConflict.DstDiff);
        }

        void IMergeViewMenuOperations.CopyFilePath(bool relativePath)
        {
            EditorGUIUtility.systemCopyBuffer = GetFilePathList.FromMergeChangeInfos(
                mMergeTreeView.GetSelectedMergeChanges(),
                relativePath,
                mWkInfo.ClientPath);
        }

        void MergeViewFileConflictMenu.IMetaMenuOperations.ShowHistory()
        {
            MergeChangeInfo mergeChangeInfo = MergeSelection.GetSingleSelectedMergeChange(mMergeTreeView);
            MergeChangeInfo metaChangeInfo = mMergeTreeView.GetMetaChange(mergeChangeInfo);
            RevisionInfo revInfo = mergeChangeInfo.GetRevision();

            mHistoryViewLauncher.ShowHistoryView(
                metaChangeInfo.GetMount().RepSpec,
                revInfo.ItemId,
                metaChangeInfo.GetPath(),
                revInfo.Type == EnumRevisionType.enDirectory);
        }

        void MergeViewFileConflictMenu.IMetaMenuOperations.DiffDestinationWithAncestor()
        {
            MergeChangeInfo mergeChange = MergeSelection.
                GetSingleSelectedMergeChange(mMergeTreeView);

            if (mergeChange == null)
                return;

            DiffDestinationWithAncestorForFileConflict(
                mShowDownloadPlasticExeWindow,
                mMergeTreeView.GetMetaChange(mergeChange),
                mWkInfo);
        }

        void IMergeViewMenuOperations.ShowHistory()
        {
            MergeChangeInfo mergeChangeInfo = MergeSelection.GetSingleSelectedMergeChange(mMergeTreeView);
            RevisionInfo revInfo = mergeChangeInfo.GetRevision();

            mHistoryViewLauncher.ShowHistoryView(
                mergeChangeInfo.GetMount().RepSpec,
                revInfo.ItemId,
                mergeChangeInfo.GetPath(),
                revInfo.Type == EnumRevisionType.enDirectory);
        }

        void MergeViewFileConflictMenu.IMetaMenuOperations.DiffSourceWithAncestor()
        {
            MergeChangeInfo mergeChange = MergeSelection.
                GetSingleSelectedMergeChange(mMergeTreeView);

            if (mergeChange == null)
                return;

            DiffSourceWithAncestorForFileConflict(
                mShowDownloadPlasticExeWindow,
                mMergeTreeView.GetMetaChange(mergeChange),
                mWkInfo);
        }

        void MergeViewFileConflictMenu.IMetaMenuOperations.DiffSourceWithDestination()
        {
            MergeChangeInfo mergeChange = MergeSelection.
                GetSingleSelectedMergeChange(mMergeTreeView);

            if (mergeChange == null)
                return;

            DiffSourceWithDestinationForFileConflict(
                mShowDownloadPlasticExeWindow,
                mMergeTreeView.GetMetaChange(mergeChange),
                mWkInfo,
                mIsIncomingMerge);
        }

        bool MergeViewFileConflictMenu.IMetaMenuOperations.SelectionHasMeta()
        {
            return mMergeTreeView.SelectionHasMeta();
        }

        void SearchField_OnDownOrUpArrowKeyPressed()
        {
            mMergeTreeView.SetFocusAndEnsureSelectedItem();
        }

        void UpdateOverview(
            MergeChangesTree mergeChangesTree,
            MergeSolvedFileConflicts solvedFileConflicts)
        {
            mChangesSummary = MergeChangesTreeParser.
                GetChangesToApplySummary(mergeChangesTree);

            mFileConflictCount = MergeChangesTreeParser.GetUnsolvedFileConflictsCount(
                mergeChangesTree, solvedFileConflicts);

            mDirectoryConflictCount = MergeChangesTreeParser.GetUnsolvedDirectoryConflictsCount(
                mergeChangesTree);
        }

        void HideMessage()
        {
            mMessageLabelText = string.Empty;
            mIsMessageLabelVisible = false;
            mIsMergeFinished = false;

            mErrorMessageLabelText = string.Empty;
            mIsErrorMessageLabelVisible = false;
        }

        void DelayedClearUpdateSuccess()
        {
            mIsUpdateSuccessful = false;
        }

        void IMergeViewMenuOperations.ShowAnnotate()
        {
        }

        void AfterProcessMerges(UpdateProgress progress)
        {
            RefreshAsset.AfterLongAssetOperation(
                ProjectPackages.ShouldBeResolvedFromUpdateProgress(mWkInfo, progress));
        }

        void ShowProcessMergesButton(string processMergesButtonText)
        {
            mProcessMergesButtonText = processMergesButtonText;
            mIsProcessMergesButtonEnabled = true;
            mIsProcessMergesButtonVisible = true;
        }

        void ProcessMerges(
            List<string> selectedPaths,
            IToolLauncher toolLauncher,
            MergeContributorType contributorType)
        {
            ProcessMerges(
                mMergeViewLogic,
                contributorType,
                mWorkspaceWindow,
                mSwitcher,
                toolLauncher,
                mGuiMessage,
                selectedPaths,
                AfterProcessMerges,
                MergeSuccessfullyFinished);
        }

        void MergeSuccessfullyFinished()
        {
            mIsUpdateSuccessful = true;
        }

        void ProcessAllMerges()
        {
            ProcessMerges(
                new List<string>(),
                PlasticExeLauncher.BuildForMergeSelectedFiles(mWkInfo, false, mShowDownloadPlasticExeWindow),
                mMergeDialogParameters.Options.Contributor);
        }

        void DoConflictsTree(
            MergeTreeView mergeTreeView,
            EmptyStateData emptyStateData,
            CooldownWindowDelayer cooldownClearUpdateSuccessAction,
            bool isOperationRunning,
            bool isUpdateSuccessful,
            bool isIncomingMerge,
            Action repaint)
        {
            using (new EditorGUI.DisabledScope(isOperationRunning))
            {
                Rect rect = GUILayoutUtility.GetRect(0, 100000, 0, 100000);

                mergeTreeView.OnGUI(rect);

                if (isOperationRunning)
                    return;

                if (mergeTreeView.GetTotalItemCount() == 0)
                {
                    DrawEmptyState(
                        rect,
                        emptyStateData,
                        cooldownClearUpdateSuccessAction,
                        isUpdateSuccessful,
                        isIncomingMerge,
                        repaint);
                    return;
                }

                if (isUpdateSuccessful)
                    NotifySuccessInStatusBar();
            }
        }

        void NotifySuccessInStatusBar()
        {
            mStatusBar.Notify(
                new GUIContentNotification(
                    PlasticLocalization.Name.WorkspaceUpdateCompleted.GetString()),
                UnityEditor.MessageType.None,
                Images.GetStepOkIcon());
            mIsUpdateSuccessful = false;
        }

        static void DiffSourceWithDestinationForDirectoryConflict(
            LaunchTool.IShowDownloadPlasticExeWindow showDownloadPlasticExeWindow,
            MergeChangeInfo mergeChange,
            WorkspaceInfo wkInfo,
            bool isIncomingMerge)
        {
            DirectoryConflict conflict = mergeChange.DirectoryConflict;
            MountPointWithPath mountPoint = mergeChange.GetMount();

            DiffOperation.DiffRevisions(
                wkInfo,
                mountPoint.RepSpec,
                conflict.SrcDiff.RevInfo,
                conflict.DstDiff.RevInfo,
                mountPoint.GetFullCmPath(conflict.SrcDiff.Path),
                mountPoint.GetFullCmPath(conflict.DstDiff.Path),
                isIncomingMerge,
                PlasticExeLauncher.BuildForDiffContributors(wkInfo, false, showDownloadPlasticExeWindow),
                imageDiffLauncher: null);
        }

        static void DiffDestinationWithAncestorForDirectoryConflict(
            LaunchTool.IShowDownloadPlasticExeWindow showDownloadPlasticExeWindow,
            MergeChangeInfo mergeChange,
            WorkspaceInfo wkInfo,
            bool isIncomingMerge)
        {
            DirectoryConflict conflict = mergeChange.DirectoryConflict;
            MountPointWithPath mountPoint = mergeChange.GetMount();

            DiffOperation.DiffRevisions(
                wkInfo,
                mountPoint.RepSpec,
                conflict.DstDiff.Base,
                conflict.DstDiff.RevInfo,
                mountPoint.GetFullCmPath(conflict.DstDiff.Path),
                mountPoint.GetFullCmPath(conflict.DstDiff.Path),
                isIncomingMerge,
                PlasticExeLauncher.BuildForDiffContributors(wkInfo, false, showDownloadPlasticExeWindow),
                imageDiffLauncher: null);
        }

        static void DiffSourceWithAncestorForDirectoryConflict(
            LaunchTool.IShowDownloadPlasticExeWindow showDownloadPlasticExeWindow,
            MergeChangeInfo mergeChange,
            WorkspaceInfo wkInfo,
            bool isIncomingMerge)
        {
            DirectoryConflict conflict = mergeChange.DirectoryConflict;
            MountPointWithPath mountPoint = mergeChange.GetMount();

            DiffOperation.DiffRevisions(
                wkInfo,
                mountPoint.RepSpec,
                conflict.SrcDiff.Base,
                conflict.SrcDiff.RevInfo,
                mountPoint.GetFullCmPath(conflict.SrcDiff.Path),
                mountPoint.GetFullCmPath(conflict.SrcDiff.Path),
                isIncomingMerge,
                PlasticExeLauncher.BuildForDiffContributors(wkInfo, false, showDownloadPlasticExeWindow),
                imageDiffLauncher: null);
        }

        static void DiffDestinationWithAncestorForFileConflict(
            LaunchTool.IShowDownloadPlasticExeWindow showDownloadPlasticExeWindow,
            MergeChangeInfo mergeChange,
            WorkspaceInfo wkInfo)
        {
            FileConflict conflict = mergeChange.FileConflict;
            MountPointWithPath mountPoint = mergeChange.GetMount();

            string path = mountPoint.GetFullCmPath(mergeChange.GetPath());

            DiffOperation.DiffRevisions(
                wkInfo,
                mountPoint.RepSpec,
                conflict.Base,
                conflict.DstDiff.RevInfo,
                path,
                path,
                false,
                PlasticExeLauncher.BuildForDiffContributors(wkInfo, false, showDownloadPlasticExeWindow),
                imageDiffLauncher: null);
        }

        static void DiffSourceWithDestinationForFileConflict(
            LaunchTool.IShowDownloadPlasticExeWindow showDownloadPlasticExeWindow,
            MergeChangeInfo mergeChange,
            WorkspaceInfo wkInfo,
            bool isIncomingMerge)
        {
            PlasticExeLauncher plasticExeLauncher =
                PlasticExeLauncher.BuildForDiffContributors(wkInfo, false, showDownloadPlasticExeWindow);

            if (isIncomingMerge)
            {
                DiffOperation.DiffYoursWithIncoming(
                    wkInfo,
                    mergeChange.GetMount(),
                    mergeChange.GetRevision(),
                    mergeChange.GetPath(),
                    plasticExeLauncher,
                    imageDiffLauncher: null);
                return;
            }

            FileConflict conflict = mergeChange.FileConflict;
            MountPointWithPath mountPoint = mergeChange.GetMount();

            string path = mountPoint.GetFullCmPath(mergeChange.GetPath());

            DiffOperation.DiffRevisions(
                wkInfo,
                mountPoint.RepSpec,
                conflict.SrcDiff.RevInfo,
                conflict.DstDiff.RevInfo,
                path,
                path,
                false,
                plasticExeLauncher,
                imageDiffLauncher: null);
        }

        static void UpdateFileConflictsTree(
            MergeChangesTree mergeChangesTree,
            MergeTreeView mergeTreeView)
        {
            UnityMergeTree unityMergeTree = UnityMergeTree.BuildMergeCategories(mergeChangesTree);

            mergeTreeView.BuildModel(unityMergeTree);
            mergeTreeView.Refilter();
            mergeTreeView.Sort();
            mergeTreeView.Reload();

            mergeTreeView.SelectFirstUnsolvedDirectoryConflict();
        }

        static void DiffSourceWithAncestorForFileConflict(
            LaunchTool.IShowDownloadPlasticExeWindow showDownloadPlasticExeWindow,
            MergeChangeInfo mergeChange,
            WorkspaceInfo wkInfo)
        {
            FileConflict conflict = mergeChange.FileConflict;
            MountPointWithPath mountPoint = mergeChange.GetMount();

            string path = mountPoint.GetFullCmPath(mergeChange.GetPath());

            DiffOperation.DiffRevisions(
                wkInfo,
                mountPoint.RepSpec,
                conflict.Base,
                conflict.SrcDiff.RevInfo,
                path,
                path,
                false,
                PlasticExeLauncher.BuildForDiffContributors(wkInfo, false, showDownloadPlasticExeWindow),
                imageDiffLauncher: null);
        }

        static void ProcessMerges(
            MergeViewLogic mergeViewLogic,
            MergeContributorType contributorType,
            IWorkspaceWindow workspaceWindow,
            IViewSwitcher switcher,
            IToolLauncher toolLauncher,
            GuiMessage.IGuiMessage guiMessage,
            List<string> selectedPaths,
            EndOperationDelegateForUpdateProgress afterProcessMergesAction,
            Action successOperationDelegate)
        {
            mergeViewLogic.ProcessMerges(
                workspaceWindow,
                switcher,
                guiMessage,
                selectedPaths,
                null,
                contributorType,
                toolLauncher,
                false,
                RefreshAsset.BeforeLongAssetOperation,
                afterProcessMergesAction,
                successOperationDelegate);
        }

        static void AddConflictResolution(
            MergeChangeInfo conflict,
            DirectoryConflictResolveActions resolveAction,
            string renameValue,
            List<DirectoryConflictResolutionData> conflictResolutions)
        {
            conflictResolutions.Add(new DirectoryConflictResolutionData(
                conflict.DirectoryConflict,
                conflict.Xlink,
                conflict.GetMount().Mount,
                resolveAction,
                renameValue));
        }

        static void DoTitle(
            string title,
            MergeDialogParameters mergeDialogParameters,
            MergeViewLogic mergeViewLogic)
        {
            EditorGUILayout.BeginHorizontal(EditorStyles.toolbar);

            GUILayout.Label(title, UnityStyles.MergeTab.TitleLabel);

            GUILayout.FlexibleSpace();

            if (GUILayout.Button(
                    PlasticLocalization.Name.MergeOptionsButton.GetString(),
                    EditorStyles.miniButton))
            {
                ShowMergeOptions(mergeDialogParameters, mergeViewLogic);
            }

            EditorGUILayout.EndHorizontal();
        }

        static void DrawEmptyState(
            Rect rect,
            EmptyStateData emptyStateData,
            CooldownWindowDelayer cooldownClearUpdateSuccessAction,
            bool isUpdateSuccessful,
            bool isIncomingMerge,
            Action repaint)
        {
            emptyStateData.Update(
                GetEmptyStateMessage(isUpdateSuccessful, isIncomingMerge),
                rect, Event.current.type, repaint);

            if (!isUpdateSuccessful)
            {
                DrawTreeViewEmptyState.For(emptyStateData);
                return;
            }

            if (!cooldownClearUpdateSuccessAction.IsRunning)
                cooldownClearUpdateSuccessAction.Ping();

            DrawTreeViewEmptyState.For(Images.GetStepOkIcon(), emptyStateData);
        }

        static void DoDirectoryConflictResolutionPanel(
            List<MergeChangeInfo> selectedChangeInfos,
            Action<MergeChangeInfo> resolveDirectoryConflictAction,
            Dictionary<DirectoryConflict, ConflictResolutionState> conflictResolutionStates,
            GUIContent validationLabel,
            bool isCherrypickMerge,
            bool isDirectoryResolutionPanelEnabled,
            ref ConflictResolutionState conflictResolutionState)
        {
            MergeChangeInfo selectedDirectoryConflict = selectedChangeInfos[0];

            if (selectedDirectoryConflict.DirectoryConflict.IsResolved())
                return;

            DirectoryConflictUserInfo conflictUserInfo;
            DirectoryConflictAction[] conflictActions;

            DirectoryConflictResolutionInfo.FromDirectoryConflict(
                selectedDirectoryConflict.GetMount(),
                selectedDirectoryConflict.DirectoryConflict,
                isCherrypickMerge,
                out conflictUserInfo,
                out conflictActions);

            conflictResolutionState = GetConflictResolutionState(
                selectedDirectoryConflict.DirectoryConflict,
                conflictActions,
                conflictResolutionStates);

            int pendingSelectedConflictsCount = GetPendingConflictsCount(
                selectedChangeInfos);

            DrawDirectoryResolutionPanel.ForConflict(
                selectedDirectoryConflict,
                (pendingSelectedConflictsCount <= 1) ? 0 : pendingSelectedConflictsCount - 1,
                conflictUserInfo,
                conflictActions,
                resolveDirectoryConflictAction,
                validationLabel,
                isDirectoryResolutionPanelEnabled,
                ref conflictResolutionState);
        }

        static void DoActionToolbarMessage(
            bool isMessageLabelVisible,
            string messageLabelText,
            bool isMergeFinished,
            bool isErrorMessageLabelVisible,
            string errorMessageLabelText,
            int directoryConflictCount,
            int fileConflictCount,
            MergeViewTexts.ChangesToApplySummary changesSummary,
            IViewSwitcher viewSwitcher)
        {
            if (isMergeFinished)
            {
                DoMergeAlreadyProcessedArea(viewSwitcher);
                return;
            }

            if (isMessageLabelVisible)
                DoInfoMessage(messageLabelText);

            if (isErrorMessageLabelVisible)
            {
                DoErrorMessage(errorMessageLabelText);
            }

            if (!isMessageLabelVisible && !isErrorMessageLabelVisible)
            {
                DrawMergeOverview.For(
                    directoryConflictCount,
                    fileConflictCount,
                    changesSummary);
            }
        }

        static void ShowMergeOptions(
            MergeDialogParameters mergeDialogParameters,
            MergeViewLogic mergeViewLogic)
        {
            bool previousMergeTrackingValue =
                mergeDialogParameters.Options.IgnoreMergeTracking;

            ChangesetSpec previousAncestor = mergeDialogParameters.AncestorSpec;
            bool bIsPrevManualStrategy = mergeDialogParameters.Strategy == MergeStrategy.Manual;

            if (!MergeOptionsDialog.MergeOptions(mergeDialogParameters))
                return;

            if (!MergeDialogParameters.AreParametersChanged(
                    previousMergeTrackingValue,
                    previousAncestor,
                    bIsPrevManualStrategy,
                    mergeDialogParameters))
                return;

            mergeViewLogic.Refresh();
        }

        static void DoProcessMergesButton(
            bool isEnabled,
            string processMergesButtonText,
            IViewSwitcher switcher,
            LaunchTool.IShowDownloadPlasticExeWindow showDownloadPlasticExeWindow,
            IWorkspaceWindow workspaceWindow,
            GuiMessage.IGuiMessage guiMessage,
            MergeViewLogic mergeViewLogic,
            MergeContributorType contributorType,
            WorkspaceInfo wkInfo,
            EndOperationDelegateForUpdateProgress afterProcessMergesAction,
            Action successOperationDelegate)
        {
            GUI.enabled = isEnabled;

            if (DrawActionButton.For(processMergesButtonText))
            {
                ProcessMerges(
                    mergeViewLogic,
                    contributorType,
                    workspaceWindow,
                    switcher,
                    PlasticExeLauncher.BuildForResolveConflicts(wkInfo, false, showDownloadPlasticExeWindow),
                    guiMessage,
                    new List<string>(),
                    afterProcessMergesAction,
                    successOperationDelegate);
            }

            GUI.enabled = true;
        }

        static void DoDiscardChangesButton(
            WorkspaceInfo wkInfo,
            ChangesetInfo automaticShelvesetToDiscard,
            CheckShelvedChanges.IUpdateShelvedChangesNotification updateShelvedChangesNotification,
            IShelvedChangesUpdater shelvedChangesUpdater,
            IRefreshView refreshView,
            IViewSwitcher viewSwitcher)
        {
            if (DrawActionButton.For(PlasticLocalization.Name.DiscardChangesButton.GetString()))
            {
                ShelvedChangesNotificationPanelOperations.DiscardShelvedChanges(
                    wkInfo,
                    automaticShelvesetToDiscard,
                    updateShelvedChangesNotification,
                    shelvedChangesUpdater,
                    viewSwitcher,
                    refreshView);
            }
        }

        static bool DoCancelMergesButton(
            bool isEnabled,
            MergeViewLogic mergeViewLogic)
        {
            bool shouldCancelMergesButtonEnabled = true;

            GUI.enabled = isEnabled;

            if (DrawActionButton.For(PlasticLocalization.GetString(
                    PlasticLocalization.Name.CancelButton)))
            {
                mergeViewLogic.Cancel();

                shouldCancelMergesButtonEnabled = false;
            }

            GUI.enabled = true;

            return shouldCancelMergesButtonEnabled;
        }

        static void DoWarningMessage()
        {
            string label = PlasticLocalization.GetString(PlasticLocalization.Name.SolveConflictsInLable);

            GUILayout.Label(
                new GUIContent(label, Images.GetWarnIcon()),
                UnityStyles.HeaderWarningLabel);
        }

        static void DoInfoMessage(string message)
        {
            EditorGUILayout.BeginHorizontal();

            GUILayout.Label(message, UnityStyles.MergeTab.InfoLabel);

            EditorGUILayout.EndHorizontal();
        }

        static void DoMergeAlreadyProcessedArea(IViewSwitcher viewSwitcher)
        {
            EditorGUILayout.BeginHorizontal();

            GUILayout.Label(
                PlasticLocalization.Name.MergeAlreadyProcessedOpenPendingChangesText.GetString(),
                UnityStyles.MergeTab.InfoLabel);

            if (GUILayout.Button(
                PlasticLocalization.Name.MergeAlreadyProcessedOpenPendingChangesLinkText.GetString(),
                UnityStyles.MergeTab.LinkLabel))
            {
                viewSwitcher.ShowPendingChanges();
                viewSwitcher.DisableMergeView();
            }

            EditorGUIUtility.AddCursorRect(
                GUILayoutUtility.GetLastRect(), MouseCursor.Link);

            GUILayout.Label(
                PlasticLocalization.Name.MergeAlreadyProcessedOpenPreferencesText.GetString(),
                UnityStyles.MergeTab.InfoLabel);

            if (GUILayout.Button(
                PlasticLocalization.Name.MergeAlreadyProcessedOpenPreferencesLinkText.GetString(),
                UnityStyles.MergeTab.LinkLabel))
            {
                OpenPlasticProjectSettings.InDiffAndMergeFoldout();
            }

            EditorGUIUtility.AddCursorRect(
                GUILayoutUtility.GetLastRect(), MouseCursor.Link);

            EditorGUILayout.EndHorizontal();
        }

        static void DoErrorMessage(string message)
        {
            EditorGUILayout.BeginHorizontal();

            GUILayout.Label(message, UnityStyles.MergeTab.RedPendingConflictsOfTotalLabel);

            EditorGUILayout.EndHorizontal();
        }

        static ConflictResolutionState GetConflictResolutionState(
            DirectoryConflict directoryConflict,
            DirectoryConflictAction[] conflictActions,
            Dictionary<DirectoryConflict, ConflictResolutionState> conflictResoltionStates)
        {
            ConflictResolutionState result;

            if (conflictResoltionStates.TryGetValue(directoryConflict, out result))
                return result;

            result = ConflictResolutionState.Build(directoryConflict, conflictActions);

            conflictResoltionStates.Add(directoryConflict, result);
            return result;
        }

        static string GetEmptyStateMessage(
            bool isUpdateSuccessful,
            bool isIncomingMerge)
        {
            if (isUpdateSuccessful)
                return PlasticLocalization.Name.WorkspaceUpdateCompleted.GetString();

            return isIncomingMerge ?
                PlasticLocalization.Name.NoIncomingChanges.GetString() :
                PlasticLocalization.Name.NoMergeChanges.GetString();
        }

        static int GetPendingConflictsCount(
            List<MergeChangeInfo> selectedChangeInfos)
        {
            int result = 0;
            foreach (MergeChangeInfo changeInfo in selectedChangeInfos)
            {
                if (changeInfo.DirectoryConflict.IsResolved())
                    continue;

                result++;
            }

            return result;
        }

        void BuildComponents(
            WorkspaceInfo wkInfo,
            bool isIncomingMerge,
            bool isMergeTo,
            bool isShelvesetMerge)
        {
            mSearchField = new SearchField();
            mSearchField.downOrUpArrowKeyPressed +=
                SearchField_OnDownOrUpArrowKeyPressed;

            MergeTreeHeaderState mergeHeaderState =
                MergeTreeHeaderState.GetDefault();

            TreeHeaderSettings.Load(mergeHeaderState,
                isIncomingMerge ?
                    UnityConstants.DEVELOPER_INCOMING_CHANGES_TABLE_SETTINGS_NAME :
                    UnityConstants.DEVELOPER_MERGE_TABLE_SETTINGS_NAME,
                (int)MergeTreeColumn.Path, true);

            mMenu = new MergeViewMenu(this, this, isIncomingMerge, isMergeTo, isShelvesetMerge);
            mMergeTreeView = new MergeTreeView(
                wkInfo, mergeHeaderState,
                MergeTreeHeaderState.GetColumnNames(),
                mMenu);

            mMergeTreeView.Reload();
        }

        bool mIsProcessMergesButtonVisible;
        bool mIsCancelMergesButtonVisible;
        bool mIsMessageLabelVisible;
        bool mIsErrorMessageLabelVisible;
        bool mIsMergeFinished;

        bool mIsProcessMergesButtonEnabled;
        bool mIsCancelMergesButtonEnabled;
        bool mIsDirectoryResolutionPanelEnabled = true;
        bool mHasPendingDirectoryConflicts;
        bool mIsOperationRunning;
        bool mIsUpdateSuccessful;

        string mProcessMergesButtonText;

        string mTitleText;
        string mMessageLabelText;
        string mErrorMessageLabelText;

        SearchField mSearchField;
        MergeTreeView mMergeTreeView;

        MergeChangesTree mMergeChangesTree;
        MergeViewMenu mMenu;
        MergeDialogParameters mMergeDialogParameters;

        Dictionary<DirectoryConflict, ConflictResolutionState> mConflictResolutionStates =
            new Dictionary<DirectoryConflict, ConflictResolutionState>();

        ConflictResolutionState mConflictResolutionState;

        int mDirectoryConflictCount;
        int mFileConflictCount;
        MergeViewTexts.ChangesToApplySummary mChangesSummary;
        readonly GUIContent mValidationLabel = new GUIContent(string.Empty, Images.GetWarnIcon());
        readonly EmptyStateData mEmptyStateData = new EmptyStateData();
        readonly ProgressControlsForViews mProgressControls;
        readonly CooldownWindowDelayer mCooldownClearUpdateSuccessAction;
        readonly MergeViewLogic mMergeViewLogic;
        readonly RepositorySpec mRepSpec;
        readonly bool mIsMergeTo;
        readonly bool mShowDiscardChangesButton;
        readonly bool mIsIncomingMerge;
        readonly IRefreshView mRefreshView;
        readonly IShelvedChangesUpdater mShelvedChangesUpdater;
        readonly CheckShelvedChanges.IUpdateShelvedChangesNotification mUpdateShelvedChangesNotification;
        readonly MergeViewLogic.IMergeController mMergeController;
        readonly GuiMessage.IGuiMessage mGuiMessage;
        readonly ObjectInfo mObjectInfo;
        readonly ObjectInfo mAncestorObjectInfo;
        readonly EnumMergeType mMergeType;
        readonly ShowIncomingChangesFrom mFrom;
        readonly EditorWindow mParentWindow;
        readonly StatusBar mStatusBar;
        readonly NewIncomingChangesUpdater mNewIncomingChangesUpdater;
        readonly IHistoryViewLauncher mHistoryViewLauncher;
        readonly LaunchTool.IShowDownloadPlasticExeWindow mShowDownloadPlasticExeWindow;
        readonly IViewSwitcher mSwitcher;
        readonly IWorkspaceWindow mWorkspaceWindow;
        readonly WorkspaceInfo mWkInfo;
    }
}
