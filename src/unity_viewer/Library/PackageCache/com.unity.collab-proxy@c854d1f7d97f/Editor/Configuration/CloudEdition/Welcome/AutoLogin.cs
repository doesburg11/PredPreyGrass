using System;
using UnityEditor;
using UnityEngine;

using Codice.Client.Common.Threading;
using Codice.CM.Common;
using Codice.LogWrapper;
using PlasticGui;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.WebApi;

namespace Unity.PlasticSCM.Editor.Configuration.CloudEdition.Welcome
{
    internal class AutoLogin
    {
        internal enum State : byte
        {
            Off = 0,
            Started = 1,
            Running = 2,
            ResponseInit = 3,
            ResponseEnd = 4,
            ResponseSuccess = 5,
            ErrorNoToken = 20,
            ErrorTokenException = 21,
            ErrorResponseNull = 22,
            ErrorResponseError = 23,
            ErrorTokenEmpty = 24,
        }

        internal void Run()
        {
            mPlasticWindow = GetPlasticWindow();

            if (!string.IsNullOrEmpty(CloudProjectSettings.accessToken))
            {
                mLog.Debug("Run");
                ExchangeTokensAndJoinOrganizationInThreadWaiter(CloudProjectSettings.accessToken);
                return;
            }

            mPlasticWindow.GetWelcomeView().autoLoginState = AutoLogin.State.ErrorNoToken;
        }

        void ExchangeTokensAndJoinOrganizationInThreadWaiter(string unityAccessToken)
        {
            int ini = Environment.TickCount;

            TokenExchangeResponse tokenExchangeResponse = null;

            IThreadWaiter waiter = ThreadWaiter.GetWaiter(10);
            waiter.Execute(
            /*threadOperationDelegate*/ delegate
            {
                mPlasticWindow.GetWelcomeView().autoLoginState = AutoLogin.State.ResponseInit;
                tokenExchangeResponse = WebRestApiClient.PlasticScm.TokenExchange(unityAccessToken);
            },
            /*afterOperationDelegate*/ delegate
            {
                mLog.DebugFormat(
                    "TokenExchange time {0} ms",
                    Environment.TickCount - ini);

                if (waiter.Exception != null)
                {
                    mPlasticWindow.GetWelcomeView().autoLoginState = AutoLogin.State.ErrorTokenException;
                    ExceptionsHandler.LogException(
                        "TokenExchangeSetting",
                        waiter.Exception);
                    Debug.LogWarning(waiter.Exception.Message);
                    return;
                }

                if (tokenExchangeResponse == null)
                {
                    mPlasticWindow.GetWelcomeView().autoLoginState = AutoLogin.State.ErrorResponseNull;
                    var warning = PlasticLocalization.GetString(PlasticLocalization.Name.TokenExchangeResponseNull);
                    mLog.Warn(warning);
                    Debug.LogWarning(warning);
                    return;
                }

                if (tokenExchangeResponse.Error != null)
                {
                    mPlasticWindow.GetWelcomeView().autoLoginState = AutoLogin.State.ErrorResponseError;
                    var warning = string.Format(
                        PlasticLocalization.GetString(PlasticLocalization.Name.TokenExchangeResponseError),
                        tokenExchangeResponse.Error.Message, tokenExchangeResponse.Error.ErrorCode);
                    mLog.ErrorFormat(warning);
                    Debug.LogWarning(warning);
                    return;
                }

                if (string.IsNullOrEmpty(tokenExchangeResponse.AccessToken))
                {
                    mPlasticWindow.GetWelcomeView().autoLoginState = AutoLogin.State.ErrorTokenEmpty;
                    var warning = string.Format(
                        PlasticLocalization.GetString(PlasticLocalization.Name.TokenExchangeAccessEmpty),
                        tokenExchangeResponse.User);
                    mLog.InfoFormat(warning);
                    Debug.LogWarning(warning);
                    return;
                }

                mPlasticWindow.GetWelcomeView().autoLoginState = AutoLogin.State.ResponseEnd;

                Credentials credentials = new Credentials(
                    new SEID(tokenExchangeResponse.User, false, tokenExchangeResponse.AccessToken),
                    SEIDWorkingMode.SSOWorkingMode);

                ShowOrganizationsPanel(credentials);
            });
        }

        void ShowOrganizationsPanel(Credentials credentials)
        {
            mPlasticWindow = GetPlasticWindow();
            mPlasticWindow.GetWelcomeView().autoLoginState = AutoLogin.State.ResponseSuccess;

            CloudEditionWelcomeWindow.ShowWindow(
                PlasticGui.Plastic.WebRestAPI, null, true);

            mCloudEditionWelcomeWindow = CloudEditionWelcomeWindow.GetWelcomeWindow();

            mCloudEditionWelcomeWindow.GetOrganizations(credentials);

            mCloudEditionWelcomeWindow.Focus();
        }

        static PlasticWindow GetPlasticWindow()
        {
            var windows = Resources.FindObjectsOfTypeAll<PlasticWindow>();
            PlasticWindow plasticWindow = windows.Length > 0 ? windows[0] : null;

            if (plasticWindow == null)
                plasticWindow = ShowWindow.Plastic();

            return plasticWindow;
        }

        PlasticWindow mPlasticWindow;
        CloudEditionWelcomeWindow mCloudEditionWelcomeWindow;

        static readonly ILog mLog = PlasticApp.GetLogger("AutoLogin");
    }
}
