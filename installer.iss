; Inno Setup script for WhisperType.
; Per-user installer (no admin/UAC). The Whisper model and the optional GPU
; libraries are downloaded during installation, so the installer stays small.

#define MyAppName "WhisperType"
#define MyAppVersion "1.2.0"
#define MyAppPublisher "WhisperType"
#define MyAppExeName "WhisperType.exe"
#define MyFetchExeName "WhisperTypeFetch.exe"
; Folder produced by PyInstaller. Overridable with ISCC /DDistDir=...
#ifndef DistDir
  #define DistDir "C:\WhisperTypeBuild\slimdist\WhisperType"
#endif
; Where the finished installer is written. Overridable with ISCC /DOutDir=...
#ifndef OutDir
  #define OutDir "C:\WhisperTypeBuild\Output"
#endif

[Setup]
AppId={{9F2A7C4E-1B3D-4E6A-9C88-WHISPERTYPE01}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir={#OutDir}
OutputBaseFilename=WhisperType-Setup
SetupIconFile=icon.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; Flags: unchecked
Name: "startupicon"; Description: "Start {#MyAppName} automatically when I sign in to Windows"; GroupDescription: "Startup:"

[Files]
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; \
    ValueType: string; ValueName: "{#MyAppName}"; ValueData: """{app}\{#MyAppExeName}"""; \
    Flags: uninsdeletevalue; Tasks: startupicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName} now"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Components downloaded after install (models + CUDA libraries).
Type: filesandordirs; Name: "{localappdata}\{#MyAppName}\models"
Type: filesandordirs; Name: "{localappdata}\{#MyAppName}\nvidia"

[Code]
var
  ChoicePage: TWizardPage;
  ModelCombo: TNewComboBox;
  GpuCheck: TNewCheckBox;
  SizeLabel: TNewStaticText;

function ModelId(Index: Integer): String;
begin
  case Index of
    0: Result := 'base.en';
    1: Result := 'small.en';
    2: Result := 'medium.en';
    3: Result := 'large-v3';
  else
    Result := 'small.en';
  end;
end;

procedure UpdateSizeLabel(Sender: TObject);
var
  MB: Integer;
  S: String;
begin
  case ModelCombo.ItemIndex of
    0: MB := 150;
    1: MB := 490;
    2: MB := 1500;
    3: MB := 3100;
  else
    MB := 490;
  end;
  if GpuCheck.Checked then
    MB := MB + 1400;
  if MB >= 1000 then
    S := Format('Approximate download: %.1f GB', [MB / 1024.0])
  else
    S := Format('Approximate download: %d MB', [MB]);
  SizeLabel.Caption := S;
end;

procedure InitializeWizard;
begin
  ChoicePage := CreateCustomPage(wpSelectTasks,
    'Speech model and acceleration',
    'Choose what to download during installation.');

  with TNewStaticText.Create(ChoicePage) do
  begin
    Parent := ChoicePage.Surface;
    Top := 0;
    Width := ChoicePage.SurfaceWidth;
    WordWrap := True;
    Height := 32;
    Caption := 'Larger models are more accurate but slower and bigger. ' +
               'medium.en is a good balance on a GPU.';
  end;

  with TNewStaticText.Create(ChoicePage) do
  begin
    Parent := ChoicePage.Surface;
    Top := 44;
    Caption := 'Speech model:';
  end;

  ModelCombo := TNewComboBox.Create(ChoicePage);
  ModelCombo.Parent := ChoicePage.Surface;
  ModelCombo.Top := 64;
  ModelCombo.Width := ChoicePage.SurfaceWidth;
  ModelCombo.Style := csDropDownList;
  ModelCombo.Items.Add('base.en  -  fastest, least accurate  (~150 MB)');
  ModelCombo.Items.Add('small.en  -  fast, good accuracy  (~490 MB)');
  ModelCombo.Items.Add('medium.en  -  slower, very accurate  (~1.5 GB)');
  ModelCombo.Items.Add('large-v3  -  best accuracy, multilingual  (~3.1 GB)');
  ModelCombo.ItemIndex := 2;
  ModelCombo.OnChange := @UpdateSizeLabel;

  GpuCheck := TNewCheckBox.Create(ChoicePage);
  GpuCheck.Parent := ChoicePage.Surface;
  GpuCheck.Top := 104;
  GpuCheck.Width := ChoicePage.SurfaceWidth;
  GpuCheck.Height := 34;
  GpuCheck.Caption := 'Download GPU acceleration files (~1.4 GB) - requires an NVIDIA graphics card';
  GpuCheck.Checked := True;
  GpuCheck.OnClick := @UpdateSizeLabel;

  SizeLabel := TNewStaticText.Create(ChoicePage);
  SizeLabel.Parent := ChoicePage.Surface;
  SizeLabel.Top := 146;
  SizeLabel.Width := ChoicePage.SurfaceWidth;
  SizeLabel.Font.Style := [fsBold];

  with TNewStaticText.Create(ChoicePage) do
  begin
    Parent := ChoicePage.Surface;
    Top := 172;
    Width := ChoicePage.SurfaceWidth;
    WordWrap := True;
    Height := 46;
    Caption := 'These are downloaded after the files are copied. An internet ' +
               'connection is required now; afterwards WhisperType works ' +
               'entirely offline.';
  end;

  UpdateSizeLabel(nil);
end;

function FetchArgs: String;
begin
  Result := '--fetch --model ' + ModelId(ModelCombo.ItemIndex);
  if GpuCheck.Checked then
    Result := Result + ' --cuda';
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    WizardForm.StatusLabel.Caption :=
      'Downloading speech model and components. This may take several minutes...';
    WizardForm.Refresh;
    if not Exec(ExpandConstant('{app}\{#MyFetchExeName}'), FetchArgs, '',
                SW_SHOW, ewWaitUntilTerminated, ResultCode) then
    begin
      MsgBox('Could not start the component downloader.' + #13#10 +
             'You can finish setup later by running:' + #13#10#13#10 +
             ExpandConstant('{app}\{#MyFetchExeName}') + ' ' + FetchArgs,
             mbError, MB_OK);
    end
    else if ResultCode <> 0 then
    begin
      MsgBox('The component download did not finish successfully.' + #13#10 +
             'WhisperType is installed, but will download what it needs on ' +
             'first use, or you can re-run:' + #13#10#13#10 +
             ExpandConstant('{app}\{#MyFetchExeName}') + ' ' + FetchArgs,
             mbInformation, MB_OK);
    end;
  end;
end;
