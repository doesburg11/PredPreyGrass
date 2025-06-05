using System.Reflection;

using Codice.LogWrapper;

using PackageManager = UnityEditor.PackageManager;

namespace Unity.PlasticSCM.Editor
{
    internal static class UVCPackageVersion
    {
        internal static string Value { get; private set; }

        internal static void Initialize()
        {
            Assembly assembly = Assembly.GetExecutingAssembly();

            Value = FindPackageVersion(assembly);
        }

        static string FindPackageVersion(Assembly assembly)
        {
            PackageManager.PackageInfo packageInfo =
                PackageManager.PackageInfo.FindForAssembly(assembly);

            if (packageInfo == null)
            {
                mLog.DebugFormat("No package found for {0} (dev env plugin)", assembly);
                return "0.0.0";
            }

            string result = packageInfo.version;

            mLog.DebugFormat("Package version: {0}", result);
            return result;
        }

        static readonly ILog mLog = PlasticApp.GetLogger("UVCPackageVersion");
    }
}
