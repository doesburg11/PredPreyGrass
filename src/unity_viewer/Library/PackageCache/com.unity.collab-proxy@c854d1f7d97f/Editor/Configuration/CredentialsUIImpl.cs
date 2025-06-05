using System;
using System.Threading.Tasks;

using UnityEditor;

using Codice.CM.Common;
using Codice.Client.Common;
using Codice.Client.Common.Connection;
using Codice.Client.Common.Threading;
using PlasticGui;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.WebApi;

namespace Unity.PlasticSCM.Editor.Configuration
{
    internal class CredentialsUiImpl : AskCredentialsToUser.IGui
    {
        AskCredentialsToUser.DialogData AskCredentialsToUser.IGui.AskUserForCredentials(
            string servername, SEIDWorkingMode seidWorkingMode)
        {
            if (!PlasticPlugin.ConnectionMonitor.IsConnected)
                return AskCredentialsToUser.DialogData.Failure(seidWorkingMode);

            if (seidWorkingMode == SEIDWorkingMode.OIDCWorkingMode
                || seidWorkingMode == SEIDWorkingMode.SAMLWorkingMode)
            {
                throw new NotImplementedException(
                    string.Format("Authentication for {0} not supported yet.", seidWorkingMode));
            }

            AskCredentialsToUser.DialogData result = null;

            GUIActionRunner.RunGUIAction(delegate
            {
                result = seidWorkingMode == SEIDWorkingMode.SSOWorkingMode
                    ? RunSSOCredentialsRequest(
                        servername, CloudProjectSettings.accessToken)
                    : CredentialsDialog.RequestCredentials(
                        servername, seidWorkingMode, ParentWindow.Get());
            });

            return result;
        }

        AskCredentialsToUser.DialogData RunSSOCredentialsRequest(
            string cloudServer,
            string unityAccessToken)
        {
            if (string.IsNullOrEmpty(unityAccessToken))
            {
                return SSOCredentialsDialog.RequestCredentials(
                    cloudServer, ParentWindow.Get());
            }

            TokenExchangeResponse tokenExchangeResponse =
                WaitUntilTokenExchange(unityAccessToken);

            // There is no internet connection, so no way to get credentials
            if (tokenExchangeResponse == null)
            {
                return AskCredentialsToUser.DialogData.Failure(SEIDWorkingMode.SSOWorkingMode);
            }

            if (tokenExchangeResponse.Error == null)
            {
                return AskCredentialsToUser.DialogData.Success(
                    new Credentials(
                        new SEID(
                            tokenExchangeResponse.User,
                            false,
                            tokenExchangeResponse.AccessToken),
                        SEIDWorkingMode.SSOWorkingMode));
            }

            return SSOCredentialsDialog.RequestCredentials(
                cloudServer, ParentWindow.Get());
        }

        static TokenExchangeResponse WaitUntilTokenExchange(
            string unityAccessToken)
        {
            TokenExchangeResponse result = null;

            Task.Run(() =>
            {
                try
                {
                    result = WebRestApiClient
                        .PlasticScm
                        .TokenExchange(unityAccessToken);
                }
                catch (Exception ex)
                {
                    ExceptionsHandler.LogException(
                        "CredentialsUiImpl", ex);
                }
            }).Wait();

            return result;
        }
    }
}
