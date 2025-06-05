using System;

using UnityEditor;
using UnityEngine;

using Codice.Client.Common.Authentication;
using PlasticGui;

namespace Unity.PlasticSCM.Editor.UI.Progress
{
    class ProgressControlsForDialogs : IProgressControls, IAuthenticationProgressControls
    {
        internal class Data
        {
            internal bool IsWaitingAsyncResult;
            internal float ProgressPercent;
            internal string ProgressMessage;

            internal MessageType StatusType;
            internal string StatusMessage;

            internal void CopyInto(Data other)
            {
                other.IsWaitingAsyncResult = IsWaitingAsyncResult;
                other.ProgressPercent = ProgressPercent;
                other.ProgressMessage = ProgressMessage;
                other.StatusType = StatusType;
                other.StatusMessage = StatusMessage;
            }
        }

        internal Data ProgressData { get { return mData; } }

        internal void ForcedUpdateProgress(EditorWindow dialog)
        {
            double updateTime;
            float progressPercent;
            GetUpdateProgress(
                mLastUpdateTime, mData.ProgressPercent,
                out updateTime, out progressPercent);

            mLastUpdateTime = updateTime;

            if (!mData.IsWaitingAsyncResult)
                return;

            mData.ProgressPercent = progressPercent;

            if (Event.current.type == EventType.Repaint)
                dialog.Repaint();
        }

        void IProgressControls.HideProgress()
        {
            InternalHideProgress();
        }

        void IAuthenticationProgressControls.HideProgress()
        {
            InternalHideProgress();
        }

        void InternalHideProgress()
        {
            mData.IsWaitingAsyncResult = false;
            mData.ProgressMessage = string.Empty;
        }

        void IProgressControls.ShowProgress(string message)
        {
            InternalShowProgress(message);
        }

        void IAuthenticationProgressControls.ShowProgress(string message)
        {
            InternalShowProgress(message);
        }

        void InternalShowProgress(string message)
        {
            CleanStatusMessage(mData);

            mData.IsWaitingAsyncResult = true;
            mData.ProgressPercent = 0f;
            mData.ProgressMessage = message;
        }

        void IProgressControls.ShowError(string message)
        {
            mData.StatusMessage = message;
            mData.StatusType = MessageType.Error;
        }

        void IProgressControls.ShowNotification(string message)
        {
            mData.StatusMessage = message;
            mData.StatusType = MessageType.Info;
        }

        void IProgressControls.ShowSuccess(string message)
        {
            mData.StatusMessage = message;
            mData.StatusType = MessageType.Info;
        }

        void IProgressControls.ShowWarning(string message)
        {
            mData.StatusMessage = message;
            mData.StatusType = MessageType.Warning;
        }

        static void CleanStatusMessage(Data data)
        {
            data.StatusMessage = string.Empty;
            data.StatusType = MessageType.None;
        }

        static void GetUpdateProgress(
            double lastUpdateTime, float lastProgressPercent,
            out double updateTime, out float progressPercent)
        {
            updateTime = EditorApplication.timeSinceStartup;

            double deltaTime = Math.Min(0.1, updateTime - lastUpdateTime);
            double deltaPercent = (deltaTime / 0.1) * PERCENT_PER_SECONDS;

            progressPercent = Mathf.Repeat(
                lastProgressPercent + (float)deltaPercent, 1f);
        }

        double mLastUpdateTime = 0.0;

        Data mData = new Data();

        const double PERCENT_PER_SECONDS = 0.06;
    }
}