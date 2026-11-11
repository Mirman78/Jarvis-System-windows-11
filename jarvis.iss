#define MyAppName "J.A.R.V.I.S."
#define MyAppVersion "5.0.0"
#define MyAppExe "J.A.R.V.I.S.exe"
#define BuildDir "dist\win-unpacked"

[Setup]
AppId={{B7C1E5A2-4D3F-4A8E-9C21-6F0A1D2E3B45}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={localappdata}\Programs\JARVIS
DefaultGroupName={#MyAppName}
PrivilegesRequired=lowest
OutputDir=installer
OutputBaseFilename=JARVIS-Setup-{#MyAppVersion}
UninstallDisplayIcon={app}\{#MyAppExe}
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; Flags: unchecked

[Files]
Source: "{#BuildDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; GGUF models (optional) - never compress these:
; Source: "models\*.gguf"; DestDir: "{app}\resources\models"; Flags: nocompression ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExe}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[Code]
function PythonInstalled: Boolean;
var
  ResultCode: Integer;
begin
  Result := Exec('cmd.exe', '/c py -3 --version', '', SW_HIDE,
                 ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
  if not Result then
    Result := Exec('cmd.exe', '/c python --version', '', SW_HIDE,
                   ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ErrCode: Integer;
begin
  if (CurStep = ssPostInstall) and (not PythonInstalled) then
  begin
    if MsgBox('Python 3 was not found. J.A.R.V.I.S. needs it for the AI server.' + #13#10 +
              'Open the Python download page now?', mbConfirmation, MB_YESNO) = IDYES then
      ShellExec('open', 'https://www.python.org/downloads/', '', '', SW_SHOW, ewNoWait, ErrCode);
  end;
end;