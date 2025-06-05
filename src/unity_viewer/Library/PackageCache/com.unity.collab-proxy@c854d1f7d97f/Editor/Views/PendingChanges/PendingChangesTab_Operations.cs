using System;
using System.Collections.Generic;
using System.Linq;

using Codice.Client.BaseCommands;
using Codice.Client.Common;
using Codice.Client.Common.EventTracking;
using Codice.Client.Common.Threading;
using Codice.CM.Common;
using GluonGui.WorkspaceWindow.Views.Checkin.Operations;
using PlasticGui;
using Unity.PlasticSCM.Editor.AssetUtils;
using Unity.PlasticSCM.Editor.Settings;
using Unity.PlasticSCM.Editor.Views.PendingChanges.Dialogs;

using UnityEditor;

namespace Unity.PlasticSCM.Editor.Views.PendingChanges
{
    internal partial class PendingChangesTab
    {
        void UndoForMode(bool isGluonMode, bool keepLocalChanges)
        {
            List<ChangeInfo> changesToUndo;
            List<ChangeInfo> dependenciesCandidates;

            mPendingChangesTreeView.GetCheckedChanges(
                null, true,
                out changesToUndo,
                out dependenciesCandidates);

            UndoChangesForMode(
                isGluonMode, keepLocalChanges, changesToUndo, dependenciesCandidates);
        }

        void UndoChangesForMode(
            bool isGluonMode,
            bool keepLocalChanges,
            List<ChangeInfo> changesToUndo,
            List<ChangeInfo> dependenciesCandidates)
        {
            if (isGluonMode)
            {
                PartialUndoChanges(changesToUndo, dependenciesCandidates);
                return;
            }

            UndoChanges(changesToUndo, dependenciesCandidates, keepLocalChanges);
        }

        void UndoUnchanged()
        {
            List<ChangeInfo> changesToUndo;
            List<ChangeInfo> dependenciesCandidates;

            mPendingChangesTreeView.GetCheckedChanges(
                null, true,
                out changesToUndo,
                out dependenciesCandidates);

            UndoUnchangedChanges(changesToUndo);
        }

        void CheckinForMode(
            bool isGluonMode,
            bool keepItemsLocked)
        {
            List<ChangeInfo> changesToCheckin;
            List<ChangeInfo> dependenciesCandidates;

            mPendingChangesTreeView.GetCheckedChanges(
                null, false,
                out changesToCheckin,
                out dependenciesCandidates);

            CheckinChangesForMode(
                changesToCheckin, dependenciesCandidates, isGluonMode, keepItemsLocked);
        }

        void CheckinChangesForMode(
            List<ChangeInfo> changesToCheckin,
            List<ChangeInfo> dependenciesCandidates,
            bool isGluonMode,
            bool keepItemsLocked)
        {
            if (isGluonMode)
            {
                PartialCheckinChanges(changesToCheckin, dependenciesCandidates, keepItemsLocked);
                return;
            }

            CheckinChanges(changesToCheckin, dependenciesCandidates);
        }

        void ShelveForMode(
            bool isGluonMode,
            bool keepItemsLocked)
        {
            List<ChangeInfo> changesToShelve;
            List<ChangeInfo> dependenciesCandidates;

            mPendingChangesTreeView.GetCheckedChanges(
                null, false,
                out changesToShelve,
                out dependenciesCandidates);

            ShelveChangesForMode(
                changesToShelve, dependenciesCandidates, isGluonMode, keepItemsLocked);
        }

        void ShelveChangesForMode(
           List<ChangeInfo> changesToShelve,
           List<ChangeInfo> dependenciesCandidates,
           bool isGluonMode,
           bool keepItemsLocked)
        {
            if (isGluonMode)
            {
                PartialCheckinChanges(
                    changesToShelve, dependenciesCandidates, keepItemsLocked, isShelve: true);
                return;
            }

            ShelveChanges(changesToShelve, dependenciesCandidates);
        }

        void PartialCheckinChanges(
            List<ChangeInfo> changesToCheckin,
            List<ChangeInfo> dependenciesCandidates,
            bool keepItemsLocked,
            bool isShelve = false)
        {
            if (CheckEmptyOperation(changesToCheckin))
            {
                ((IProgressControls)mProgressControls).ShowWarning(
                    PlasticLocalization.GetString(PlasticLocalization.Name.NoItemsAreSelected));
                return;
            }

            bool isCancelled;
            mSaveAssets.ForChangesWithConfirmation(
                mWkInfo.ClientPath, changesToCheckin, mWorkspaceOperationsMonitor,
                out isCancelled);

            if (isCancelled)
                return;

            CheckinUIOperation checkinOperation = new CheckinUIOperation(
                mWkInfo,
                mViewHost,
                mProgressControls,
                mGuiMessage,
                new LaunchCheckinConflictsDialog(mParentWindow),
                new LaunchDependenciesDialog(
                    PlasticLocalization.GetString(PlasticLocalization.Name.CheckinButton),
                    mParentWindow),
                this,
                mWorkspaceWindow.GluonProgressOperationHandler,
                null);

            checkinOperation.Checkin(
                changesToCheckin,
                dependenciesCandidates,
                mCommentText,
                keepItemsLocked,
                isShelve,
                isShelve ?
                    OpenPlasticProjectSettings.InShelveAndSwitchFoldout:
                    (Action)null,
                isShelve ?
                    GetUndoEndOperationDelegate(changesToCheckin, false) :
                    (Action)null,
                isShelve ?
                    (Action)null :
                    RefreshAsset.UnityAssetDatabase,
                GetSuccessOperationDelegateForCreatedChangeset(
                    isShelve ?
                        CreatedChangesetData.Type.Shelve :
                        CreatedChangesetData.Type.Checkin));
        }

        void CheckinChanges(
            List<ChangeInfo> changesToCheckin,
            List<ChangeInfo> dependenciesCandidates)
        {
            if (CheckEmptyOperation(changesToCheckin, HasPendingMergeLinks()))
            {
                ((IProgressControls)mProgressControls).ShowWarning(
                    PlasticLocalization.GetString(PlasticLocalization.Name.NoItemsAreSelected));
                return;
            }

            bool isCancelled;
            mSaveAssets.ForChangesWithConfirmation(
                mWkInfo.ClientPath, changesToCheckin, mWorkspaceOperationsMonitor,
                out isCancelled);

            if (isCancelled)
                return;

            mPendingChangesOperations.Checkin(
                changesToCheckin,
                dependenciesCandidates,
                mCommentText,
                null,
                RefreshAsset.UnityAssetDatabase,
                GetSuccessOperationDelegateForCreatedChangeset(CreatedChangesetData.Type.Checkin));
        }

        void ShelveChanges(
            List<ChangeInfo> changesToShelve,
            List<ChangeInfo> dependenciesCandidates)
        {
            bool hasPendingMergeLinks = HasPendingMergeLinks();

            if (hasPendingMergeLinks &&
                !UserWantsShelveWithPendingMergeLinks(mGuiMessage))
            {
                return;
            }

            if (CheckEmptyOperation(changesToShelve, hasPendingMergeLinks))
            {
                ((IProgressControls)mProgressControls).ShowWarning(
                    PlasticLocalization.GetString(PlasticLocalization.Name.NoItemsAreSelected));
                return;
            }

            bool isCancelled;
            mSaveAssets.ForChangesWithConfirmation(
                mWkInfo.ClientPath, changesToShelve, mWorkspaceOperationsMonitor,
                out isCancelled);

            if (isCancelled)
                return;

            mPendingChangesOperations.Shelve(
                changesToShelve,
                dependenciesCandidates,
                mCommentText,
                OpenPlasticProjectSettings.InShelveAndSwitchFoldout,
                GetUndoEndOperationDelegate(changesToShelve, false),
                null,
                GetSuccessOperationDelegateForCreatedChangeset(CreatedChangesetData.Type.Shelve));
        }

        void PartialUndoChanges(
            List<ChangeInfo> changesToUndo,
            List<ChangeInfo> dependenciesCandidates)
        {
            if (CheckEmptyOperation(changesToUndo))
            {
                ((IProgressControls)mProgressControls).ShowWarning(
                    PlasticLocalization.GetString(PlasticLocalization.Name.NoItemsToUndo));
                return;
            }

            mSaveAssets.ForChangesWithoutConfirmation(
                mWkInfo.ClientPath, changesToUndo, mWorkspaceOperationsMonitor);

            UndoUIOperation undoOperation = new UndoUIOperation(
                mWkInfo, mViewHost,
                new LaunchDependenciesDialog(
                    PlasticLocalization.GetString(PlasticLocalization.Name.UndoButton),
                    mParentWindow),
                mProgressControls);

            undoOperation.Undo(
                changesToUndo,
                dependenciesCandidates,
                RefreshAsset.UnityAssetDatabase,
                GetSuccessOperationDelegateForUndo());
        }

        void UndoChanges(
            List<ChangeInfo> changesToUndo,
            List<ChangeInfo> dependenciesCandidates,
            bool keepLocalChanges)
        {
            if (CheckEmptyOperation(changesToUndo, HasPendingMergeLinks()))
            {
                ((IProgressControls)mProgressControls).ShowWarning(
                    PlasticLocalization.GetString(PlasticLocalization.Name.NoItemsToUndo));
                return;
            }

            mSaveAssets.ForChangesWithoutConfirmation(
                mWkInfo.ClientPath, changesToUndo, mWorkspaceOperationsMonitor);

            mPendingChangesOperations.Undo(
                changesToUndo,
                dependenciesCandidates,
                mPendingMergeLinks.Count,
                keepLocalChanges,
                GetUndoEndOperationDelegate(changesToUndo, keepLocalChanges),
                GetSuccessOperationDelegateForUndo());
        }

        void UndoUnchangedChanges(List<ChangeInfo> changesToUndo)
        {
            if (CheckEmptyOperation(changesToUndo, HasPendingMergeLinks()))
            {
                ((IProgressControls) mProgressControls).ShowWarning(
                    PlasticLocalization.GetString(PlasticLocalization.Name.NoItemsToUndo));

                return;
            }

            mSaveAssets.ForChangesWithoutConfirmation(
                mWkInfo.ClientPath, changesToUndo, mWorkspaceOperationsMonitor);

            mPendingChangesOperations.UndoUnchanged(
                changesToUndo,
                RefreshAsset.UnityAssetDatabase,
                GetSuccessOperationDelegateForUndo());
        }

        void UndoCheckoutsKeepingLocalChanges()
        {
            UndoForMode(isGluonMode: false, keepLocalChanges: true);
        }

        void UndoCheckoutChangesKeepingLocalChanges(List<ChangeInfo> changesToUndo)
        {
            UndoChanges(changesToUndo, new List<ChangeInfo>(), keepLocalChanges: true);
        }

        Action GetUndoEndOperationDelegate(
            List<ChangeInfo> changesToUndo, bool keepLocalChanges)
        {
            if (keepLocalChanges)
                return null;

            return () =>
            {
                if (changesToUndo.Any(
                        change => AssetsPath.IsPackagesRootElement(change.Path) &&
                        !IsAddedChange(change)))
                {
                    mAfterOnGUIAction = RefreshAsset.UnityAssetDatabaseAndPackageManagerAsync;
                    return;
                }

                mAfterOnGUIAction = RefreshAsset.UnityAssetDatabase;
            };
        }

        void ShowShelvesView(IViewSwitcher viewSwitcher)
        {
            TrackFeatureUseEvent.For(
                mRepSpec,
                TrackFeatureUseEvent.Features.UnityPackage.ShowShelvesViewFromDropdownMenu);

            viewSwitcher.ShowShelvesView();
        }

        SuccessOperationDelegateForCreatedChangeset GetSuccessOperationDelegateForCreatedChangeset(
            CreatedChangesetData.Type operationType)
        {
            return (repSpec, createdChangesetId) =>
            {
                CreatedChangesetData createdChangesetData =
                    new CreatedChangesetData(
                        operationType,
                        createdChangesetId,
                        repSpec);

                Action openChangesetAction = () =>
                {
                    OpenChangesetId(createdChangesetData, mShowChangesetInView, mShowShelveInView);
                };

                Action copyDiffsLinkAction = () =>
                {
                    CopyDiffsLink(createdChangesetData);
                };

                mDrawOperationSuccess = new NotifySuccessForCreatedChangeset(
                    createdChangesetData,
                    openChangesetAction,
                    copyDiffsLinkAction,
                    mParentWindow.Repaint);
            };
        }

        Action GetSuccessOperationDelegateForUndo()
        {
            return () => { mDrawOperationSuccess = new NotifySuccessForUndo(mParentWindow.Repaint); };
        }

        static void OpenChangesetId(
            CreatedChangesetData data,
            IShowChangesetInView showChangesetInView,
            IShowShelveInView showShelveInView)
        {
            if (data.OperationType == CreatedChangesetData.Type.Checkin)
            {
                ShowChangesetInView(
                    showChangesetInView,
                    data.RepositorySpec,
                    data.CreatedChangesetId);
                return;
            }

            ShowShelveInView(
                showShelveInView,
                data.RepositorySpec,
                data.CreatedChangesetId);
        }

        static void CopyDiffsLink(
            CreatedChangesetData data)
        {
            EditorGUIUtility.systemCopyBuffer = GetDiffPlasticLink.FromChangesetId(
                data.RepositorySpec,
                data.CreatedChangesetId);
        }

        static void ShowShelveInView(
            IShowShelveInView showShelveInView,
            RepositorySpec repSpec,
            long shelvesetId)
        {
            ChangesetInfo shelveInfo = null;

            IThreadWaiter waiter = ThreadWaiter.GetWaiter(10);
            waiter.Execute(
                /*threadOperationDelegate*/
                delegate
                {
                    shelveInfo = PlasticGui.Plastic.API.GetChangesetInfoFromId(repSpec, shelvesetId);
                },
                /*afterOperationDelegate*/
                delegate
                {
                    if (waiter.Exception != null)
                    {
                        ExceptionsHandler.DisplayException(waiter.Exception);
                        return;
                    }

                    showShelveInView.ShowShelveInView(shelveInfo);
                });
        }

        static bool IsAddedChange(ChangeInfo change)
        {
            return ChangeTypesOperator.ContainsAny(
                change.ChangeTypes, ChangeTypesClassifier.ADDED_TYPES);
        }

        static void ShowChangesetInView(
            IShowChangesetInView showChangesetInView,
            RepositorySpec repSpec,
            long changesetId)
        {

            ChangesetInfo changesetInfo = null;

            IThreadWaiter waiter = ThreadWaiter.GetWaiter(10);
            waiter.Execute(
                /*threadOperationDelegate*/
                delegate
                {
                    changesetInfo = PlasticGui.Plastic.API.GetChangesetInfoFromId(
                        repSpec,
                        changesetId);
                },
                /*afterOperationDelegate*/
                delegate
                {
                    if (waiter.Exception != null)
                    {
                        ExceptionsHandler.DisplayException(waiter.Exception);
                        return;
                    }

                    showChangesetInView.ShowChangesetInView(changesetInfo);
                });
        }

        static bool CheckEmptyOperation(List<ChangeInfo> elements)
        {
            return elements == null || elements.Count == 0;
        }

        static bool CheckEmptyOperation(List<ChangeInfo> elements, bool bHasPendingMergeLinks)
        {
            if (bHasPendingMergeLinks)
                return false;

            if (elements != null && elements.Count > 0)
                return false;

            return true;
        }

        static bool UserWantsShelveWithPendingMergeLinks(GuiMessage.IGuiMessage guiMessage)
        {
            return guiMessage.ShowQuestion(
                PlasticLocalization.GetString(PlasticLocalization.Name.ShelveWithPendingMergeLinksRequest),
                PlasticLocalization.GetString(PlasticLocalization.Name.ShelveWithPendingMergeLinksRequestMessage),
                PlasticLocalization.GetString(PlasticLocalization.Name.ShelveButton));
        }
    }
}
