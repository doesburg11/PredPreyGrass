using System;

using Codice.Client.Common;
using Codice.CM.Common;
using PlasticGui.WorkspaceWindow;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.StatusBar;

using GluonNewIncomingChangesUpdater = PlasticGui.Gluon.WorkspaceWindow.NewIncomingChangesUpdater;
using GluonCheckIncomingChanges = PlasticGui.Gluon.WorkspaceWindow.CheckIncomingChanges;

namespace Unity.PlasticSCM.Editor
{
    internal static class NewIncomingChanges
    {
        internal static NewIncomingChangesUpdater BuildUpdaterForDeveloper(
            WorkspaceInfo wkInfo,
            ViewSwitcher viewSwitcher,
            StatusBar.IIncomingChangesNotification incomingChangesNotification,
            CheckIncomingChanges.IAutoRefreshIncomingChangesView autoRefreshIncomingChangesView,
            CheckIncomingChanges.IUpdateIncomingChanges updateIncomingChanges)
        {
            NewIncomingChangesUpdater updater = new NewIncomingChangesUpdater(
                wkInfo,
                new UnityPlasticTimerBuilder(),
                autoRefreshIncomingChangesView,
                new CheckIncomingChanges.CalculateIncomingChanges(),
                updateIncomingChanges);

            viewSwitcher.SetNewIncomingChanges(
                updater, null, incomingChangesNotification);

            updater.Start();
            return updater;
        }

        internal static GluonNewIncomingChangesUpdater BuildUpdaterForGluon(
            WorkspaceInfo wkInfo,
            ViewSwitcher viewSwitcher,
            StatusBar.IIncomingChangesNotification incomingChangesNotification,
            GluonCheckIncomingChanges.IAutoRefreshIncomingChangesView autoRefreshIncomingChangesView,
            GluonCheckIncomingChanges.IUpdateIncomingChanges updateIncomingChanges,
            GluonCheckIncomingChanges.ICalculateIncomingChanges calculateIncomingChanges)
        {
            GluonNewIncomingChangesUpdater updater = new GluonNewIncomingChangesUpdater(
                wkInfo,
                new UnityPlasticTimerBuilder(),
                autoRefreshIncomingChangesView,
                calculateIncomingChanges,
                updateIncomingChanges);

            viewSwitcher.SetNewIncomingChanges(
                null, updater, incomingChangesNotification);

            updater.Start();
            return updater;
        }

        internal static void LaunchUpdater(
            NewIncomingChangesUpdater developerNewIncomingChangesUpdater,
            GluonNewIncomingChangesUpdater gluonNewIncomingChangesUpdater)
        {
            if (developerNewIncomingChangesUpdater != null)
            {
                developerNewIncomingChangesUpdater.Start();
                developerNewIncomingChangesUpdater.Update(DateTime.Now);
            }

            if (gluonNewIncomingChangesUpdater != null)
            {
                gluonNewIncomingChangesUpdater.Start();
                gluonNewIncomingChangesUpdater.Update(DateTime.Now);
            }
        }

        internal static void StopUpdater(
            NewIncomingChangesUpdater developerNewIncomingChangesUpdater,
            GluonNewIncomingChangesUpdater gluonNewIncomingChangesUpdater)
        {
            if (developerNewIncomingChangesUpdater != null)
                developerNewIncomingChangesUpdater.Stop();

            if (gluonNewIncomingChangesUpdater != null)
                gluonNewIncomingChangesUpdater.Stop();
        }

        internal static void DisposeUpdater(
            NewIncomingChangesUpdater developerNewIncomingChangesUpdater,
            GluonNewIncomingChangesUpdater gluonNewIncomingChangesUpdater)
        {
            if (developerNewIncomingChangesUpdater != null)
                developerNewIncomingChangesUpdater.Dispose();

            if (gluonNewIncomingChangesUpdater != null)
                gluonNewIncomingChangesUpdater.Dispose();
        }
    }
}
