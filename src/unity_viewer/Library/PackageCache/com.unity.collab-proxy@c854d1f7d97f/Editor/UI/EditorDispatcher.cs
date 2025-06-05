using System;
using System.Threading;

namespace Unity.PlasticSCM.Editor.UI
{
    internal static class EditorDispatcher
    {
        internal static void InitializeMainThreadIdAndContext(
            int mainThreadId,
            SynchronizationContext mainUnitySyncContext)
        {
            mMainThreadId = mainThreadId;
            mMainUnitySyncContext = mainUnitySyncContext;
        }

        internal static bool IsOnMainThread
        {
            get { return Thread.CurrentThread.ManagedThreadId == mMainThreadId; }
        }

        internal static void Dispatch(Action task)
        {
            mMainUnitySyncContext.Post(_ => task(), null);
        }

        static SynchronizationContext mMainUnitySyncContext;
        static int mMainThreadId;
    }
}
