using System;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Collections.Generic;

using Avalonia;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Layout;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using Avalonia.Platform.Storage;

namespace NeuralNetworks_1
{
    public partial class MainWindow : Window
    {
        int inodes = 784;
        int hnodes = 100;
        int onodes = 10;

        int epochs = 1;

        double learningRate = 0.3;

        nn3S? nn3SO;

        double[] inputs =
            Array.Empty<double>();

        double[] targets =
            Array.Empty<double>();

        byte[] imagePixels =
            Array.Empty<byte>();


        string trainFile = "";

        string testFile = "";


        ObservableCollection<nodeRow> rows =
            new ObservableCollection<nodeRow>();


        public MainWindow()
        {
            InitializeComponent();

            networkDataGrid.ItemsSource =
                rows;
        }


        // ==========================================
        // CREATE NETWORK
        // ==========================================

        private void createButton_Click(
            object? sender,
            RoutedEventArgs e)
        {
            if (!int.TryParse(
                    inputTextBox.Text,
                    out inodes)
                || inodes <= 0)
            {
                statusTextBlock.Text =
                    "Invalid Input Nodes.";

                return;
            }


            if (!int.TryParse(
                    hiddenTextBox.Text,
                    out hnodes)
                || hnodes <= 0)
            {
                statusTextBlock.Text =
                    "Invalid Hidden Nodes.";

                return;
            }


            if (!int.TryParse(
                    outputTextBox.Text,
                    out onodes)
                || onodes <= 0)
            {
                statusTextBlock.Text =
                    "Invalid Output Nodes.";

                return;
            }


            string rateText =
                (learningRateTextBox.Text ?? "")
                .Replace(',', '.');


            if (!double.TryParse(
                    rateText,
                    NumberStyles.Float,
                    CultureInfo.InvariantCulture,
                    out learningRate)
                || learningRate <= 0)
            {
                statusTextBlock.Text =
                    "Invalid Learning Rate.";

                return;
            }


            nn3SO =
                new nn3S(
                    inodes,
                    hnodes,
                    onodes,
                    learningRate
                );


            targets =
                new double[onodes];


            trainButton.IsEnabled =
                true;

            queryButton.IsEnabled =
                true;


            rows.Clear();


            statusTextBlock.Text =
                $"Network created: " +
                $"{inodes}-{hnodes}-{onodes}";
        }


        // ==========================================
        // OPEN TRAIN FILE
        // ==========================================

        private async void openTrainingFileButton_Click(
            object? sender,
            RoutedEventArgs e)
        {
            var files =
                await StorageProvider
                    .OpenFilePickerAsync(
                        new FilePickerOpenOptions
                        {
                            Title =
                                "Open Training File",

                            AllowMultiple =
                                false,

                            FileTypeFilter =
                            [
                                new FilePickerFileType(
                                    "Text files")
                                {
                                    Patterns =
                                        ["*.txt"]
                                }
                            ]
                        }
                    );


            if (files.Count == 0)
            {
                return;
            }


            trainFile =
                files[0]
                    .Path
                    .LocalPath;


            statusTextBlock.Text =
                "Training file: "
                + files[0].Name;
        }


        // ==========================================
        // OPEN TEST FILE
        // ==========================================

        private async void openTestFileButton_Click(
            object? sender,
            RoutedEventArgs e)
        {
            var files =
                await StorageProvider
                    .OpenFilePickerAsync(
                        new FilePickerOpenOptions
                        {
                            Title =
                                "Open Test File",

                            AllowMultiple =
                                false,

                            FileTypeFilter =
                            [
                                new FilePickerFileType(
                                    "Text files")
                                {
                                    Patterns =
                                        ["*.txt"]
                                }
                            ]
                        }
                    );


            if (files.Count == 0)
            {
                return;
            }


            testFile =
                files[0]
                    .Path
                    .LocalPath;


            statusTextBlock.Text =
                "Test file: "
                + files[0].Name;
        }


        // ==========================================
        // READ INPUTS
        // ==========================================

        private int readInputs(
            string line)
        {
            inputs =
                new double[
                    inodes
                ];


            imagePixels =
                new byte[
                    inodes
                ];


            string[] input =
                line.Split(',');


            for (int i = 1;
                 i < input.Length
                 && i <= inodes;
                 i++)
            {
                int j =
                    i - 1;


                double pixel =
                    Convert.ToDouble(
                        input[i],
                        CultureInfo
                            .InvariantCulture
                    );


                imagePixels[j] =
                    (byte)Math.Clamp(
                        pixel,
                        0,
                        255
                    );


                inputs[j] =
                    (
                        pixel
                        * 0.99
                        / 255.0
                    )
                    + 0.01;
            }


            return
                Convert.ToInt32(
                    input[0]
                );
        }


        // ==========================================
        // TRAIN
        // ==========================================
       private async void trainButton_Click(object? sender, RoutedEventArgs e)
{
    if (nn3SO == null)
    {
        statusTextBlock.Text = "Create Network first.";
        return;
    }

    if (trainFile == "")
    {
        statusTextBlock.Text = "Open Training File first.";
        return;
    }

    if (!int.TryParse(epochsTextBox.Text, out epochs) || epochs <= 0)
        epochs = 1;

    int trainCount = 0;

    targets = new double[onodes];

    // Loss-Werte speichern
       List<int> lossIterations = new List<int>();
       List<double> lossValues = new List<double>();
    // Für Durchschnitts-Loss über jeweils 1000 Bilder
    double lossSum = 0;
    int lossCounter = 0;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        using StreamReader sr = new StreamReader(trainFile);

        string? line;

        while ((line = sr.ReadLine()) != null && line != "")
        {
            int intTarget = readInputs(line);

            // Target-Vektor erstellen
            for (int i = 0; i < onodes; i++)
            {
                targets[i] = 0.01;
            }

            targets[intTarget] = 0.99;

            // Netzwerk trainieren
            nn3SO.train(inputs, targets);

            // ------------------------------------
            // LOSS BERECHNEN
            // ------------------------------------
            double currentLoss = 0;

            for (int i = 0; i < nn3SO.Output_errors.Length; i++)
            {
                currentLoss += Math.Abs(nn3SO.Output_errors[i]);
            }

            lossSum += currentLoss;
            lossCounter++;

            trainCount++;

            // Alle 1000 Trainingsschritte
            // Durchschnitts-Loss speichern
            if (lossCounter == 1000)
            {
                double averageLoss = lossSum / lossCounter;

                lossIterations.Add(trainCount);
                lossValues.Add(averageLoss);

                lossSum = 0;
                lossCounter = 0;
            }

            // ------------------------------------
            // Recognized Number
            // ------------------------------------
            int answer = findLargestOutput(nn3SO.Final_outputs);

            recognizedTextBox.Text = answer.ToString();

            displayResults(true);

            statusTextBlock.Text =
                "Epoch " + (epoch + 1) +
                "/" + epochs +
                " - Training: " + trainCount;

            // Bild anzeigen
            if (checkBoxImage.IsChecked == true)
            {
                showImage();

                await showNextDialog(
                    "Training",
                    "Next"
                );
            }
        }
    }

    // Falls am Ende weniger als 1000 Werte übrig bleiben
    if (lossCounter > 0)
    {
        double averageLoss = lossSum / lossCounter;

        lossIterations.Add(trainCount);
        lossValues.Add(averageLoss);
    }

    // ------------------------------------
    // WEIGHT MATRICES SPEICHERN
    // ------------------------------------

    string learningRateText =
        learningRate.ToString(
            CultureInfo.InvariantCulture
        ).Replace(".", "");

    string wihFileName =
    "WIH_FASHION_60000_E" +
    epochs +
    "_LR03_H100.txt";

string whoFileName =
    "WHO_FASHION_60000_E" +
    epochs +
    "_LR03_H100.txt";

    saveMatrix(
        wihFileName,
        nn3SO.WIH
    );

    saveMatrix(
        whoFileName,
        nn3SO.WHO
    );

    // ------------------------------------
    // LOSS CSV SPEICHERN
    // ------------------------------------

    string lossFileName =
        "Loss_" +
        epochs +
        "_Epochs.csv";

    using (StreamWriter writer =
           new StreamWriter(lossFileName))
    {
        writer.WriteLine(
            "Iteration,Loss"
        );

        for (int i = 0; i < lossValues.Count; i++)
        {
            writer.WriteLine(
                lossIterations[i].ToString(
                    CultureInfo.InvariantCulture
                )
                + ","
                + lossValues[i].ToString(
                    CultureInfo.InvariantCulture
                )
            );
        }
    }

    // ------------------------------------
    // FERTIG
    // ------------------------------------

    statusTextBlock.Text =
        "Training done: " +
        trainCount +
        " - Loss saved: " +
        lossFileName;

    await showDoneDialog(
        "Training",
        "Training done: " +
        trainCount +
        "\nLoss saved in:\n" +
        lossFileName
    );
}
       

private async Task showDoneDialog(
    string title,
    string text)
{
    Window dialog =
        new Window
        {
            Title = title,
            Width = 260,
            Height = 150,
            CanResize = false,
            WindowStartupLocation =
                WindowStartupLocation.CenterOwner
        };

    Button button =
        new Button
        {
            Content = "OK",
            Width = 100,
            HorizontalAlignment =
                HorizontalAlignment.Center
        };

    button.Click +=
        (_, _) =>
        {
            dialog.Close();
        };

    dialog.Content =
        new StackPanel
        {
            Spacing = 15,
            Margin =
                new Thickness(20),

            Children =
            {
                new TextBlock
                {
                    Text = text,
                    HorizontalAlignment =
                        HorizontalAlignment.Center
                },

                button
            }
        };

    await dialog.ShowDialog(this);
}

        // ==========================================
        // QUERY
        // ==========================================

        private async void queryButton_Click(
    object? sender,
    RoutedEventArgs e)
{
    if (nn3SO == null)
    {
        statusTextBlock.Text =
            "Create or load Network first.";

        return;
    }

    if (testFile == "")
    {
        statusTextBlock.Text =
            "Open Test File first.";

        return;
    }

    int correctAnswers = 0;
    int totalAnswers = 0;

    using StreamReader sr =
        new StreamReader(testFile);

    string? line;

    while ((line = sr.ReadLine()) != null
           && line != "")
    {
        // 1. Label + Pixel lesen
        int intTarget =
            readInputs(line);


        // 2. Netzwerk abfragen
        nn3SO.queryNN(
            inputs
        );


        // 3. Antwort des Netzwerks bestimmen
        int answer =
            findLargestOutput(
                nn3SO.Final_outputs
            );


        // 4. Recognized anzeigen
        recognizedTextBox.Text =
            answer.ToString();


        // 5. Gesamtzahl erhöhen
        totalAnswers++;


        // 6. Prüfen, ob Antwort richtig ist
        if (answer == intTarget)
        {
            correctAnswers++;
        }


        // 7. Performance NACH JEDER QUERY berechnen
        double performance =
            (double)correctAnswers
            / totalAnswers;


        // 8. Performance sofort anzeigen
        performanceTextBox.Text =
            performance.ToString(
                "0.00",
                CultureInfo.InvariantCulture
            );


        // Optional: Status während des Testens
        statusTextBlock.Text =
            "Correct: "
            + correctAnswers
            + " / "
            + totalAnswers;


        // 9. DataGrid aktualisieren
        displayResults(
            false
        );


        // 10. Bild anzeigen, wenn Checkbox aktiviert ist
        if (checkBoxImage.IsChecked == true)
        {
            showImage();

            await showNextDialog(
                "Query",
                "Next"
            );
        }
    }


    // 11. Fertigmeldung
    statusTextBlock.Text =
        "Query done - Correct: "
        + correctAnswers
        + " / "
        + totalAnswers;
}


        // ==========================================
        // LARGEST OUTPUT
        // ==========================================

        private int findLargestOutput(
            double[] outputs)
        {
            int answer =
                0;


            double biggest =
                outputs[0];


            for (int i = 1;
                 i < outputs.Length;
                 i++)
            {
                if (
                    outputs[i]
                    > biggest)
                {
                    biggest =
                        outputs[i];

                    answer =
                        i;
                }
            }


            return answer;
        }


        // ==========================================
        // SHOW IMAGE
        // ==========================================

        private void showImage()
        {
            if (
                imagePixels.Length
                < 784)
            {
                return;
            }


            WriteableBitmap bitmap =
                new WriteableBitmap(
                    new PixelSize(
                        28,
                        28
                    ),
                    new Vector(
                        96,
                        96
                    ),
                    PixelFormats.Gray8,
                    AlphaFormat.Opaque
                );


            using (
                var framebuffer =
                    bitmap.Lock())
            {
                for (int y = 0;
                     y < 28;
                     y++)
                {
                    IntPtr destination =
                        IntPtr.Add(
                            framebuffer.Address,
                            y
                            * framebuffer
                                .RowBytes
                        );


                    Marshal.Copy(
                        imagePixels,
                        y * 28,
                        destination,
                        28
                    );
                }
            }


            numberImage.Source =
                bitmap;


            noImageTextBlock.IsVisible =
                false;
        }


        // ==========================================
        // NEXT DIALOG
        // ==========================================

        private async Task showNextDialog(
            string title,
            string text)
        {
            Window dialog =
                new Window
                {
                    Title = title,

                    Width = 260,

                    Height = 150,

                    CanResize =
                        false,

                    WindowStartupLocation =
                        WindowStartupLocation
                            .CenterOwner
                };


            Button button =
                new Button
                {
                    Content =
                        "Next",

                    Width =
                        100,

                    HorizontalAlignment =
                        HorizontalAlignment
                            .Center
                };


            button.Click +=
                (_, _) =>
                {
                    dialog.Close();
                };


            dialog.Content =
                new StackPanel
                {
                    Spacing = 15,

                    Margin =
                        new Thickness(
                            20
                        ),

                    Children =
                    {
                        new TextBlock
                        {
                            Text = text,

                            HorizontalAlignment =
                                HorizontalAlignment
                                    .Center
                        },

                        button
                    }
                };


            await dialog.ShowDialog(
                this
            );
        }


        // ==========================================
        // LOAD WEIGHTS
        // ==========================================

        private async void loadWeightMatricesButton_Click(
            object? sender,
            RoutedEventArgs e)
        {
            var wihFiles =
                await StorageProvider
                    .OpenFilePickerAsync(
                        new FilePickerOpenOptions
                        {
                            Title =
                                "Open WIH Matrix",

                            AllowMultiple =
                                false,

                            FileTypeFilter =
                            [
                                new FilePickerFileType(
                                    "Text files")
                                {
                                    Patterns =
                                        ["*.txt"]
                                }
                            ]
                        }
                    );


            if (wihFiles.Count == 0)
            {
                return;
            }


            var whoFiles =
                await StorageProvider
                    .OpenFilePickerAsync(
                        new FilePickerOpenOptions
                        {
                            Title =
                                "Open WHO Matrix",

                            AllowMultiple =
                                false,

                            FileTypeFilter =
                            [
                                new FilePickerFileType(
                                    "Text files")
                                {
                                    Patterns =
                                        ["*.txt"]
                                }
                            ]
                        }
                    );


            if (whoFiles.Count == 0)
            {
                return;
            }


            double[,] loadedWih =
                loadMatrix(
                    wihFiles[0]
                        .Path
                        .LocalPath
                );


            double[,] loadedWho =
                loadMatrix(
                    whoFiles[0]
                        .Path
                        .LocalPath
                );


            inodes =
                loadedWih
                    .GetLength(0);

            hnodes =
                loadedWih
                    .GetLength(1);

            onodes =
                loadedWho
                    .GetLength(1);


            nn3SO =
                new nn3S(
                    inodes,
                    hnodes,
                    onodes,
                    learningRate
                );


            nn3SO.setWIHMatrix(
                loadedWih
            );


            nn3SO.setWHOMatrix(
                loadedWho
            );


            inputTextBox.Text =
                inodes.ToString();


            hiddenTextBox.Text =
                hnodes.ToString();


            outputTextBox.Text =
                onodes.ToString();


            trainButton.IsEnabled =
                true;

            queryButton.IsEnabled =
                true;


            statusTextBlock.Text =
                "Weight matrices loaded.";
        }


        // ==========================================
        // SAVE MATRIX
        // ==========================================

        private void saveMatrix(
            string fileName,
            double[,] matrix)
        {
            using StreamWriter writer =
                new StreamWriter(
                    fileName
                );


            int rowCount =
                matrix.GetLength(0);

            int columnCount =
                matrix.GetLength(1);


            writer.WriteLine(
                rowCount
                + ","
                + columnCount
            );


            for (int i = 0;
                 i < rowCount;
                 i++)
            {
                for (int j = 0;
                     j < columnCount;
                     j++)
                {
                    if (j > 0)
                    {
                        writer.Write(",");
                    }


                    writer.Write(
                        matrix[i, j]
                            .ToString(
                                "R",
                                CultureInfo
                                    .InvariantCulture
                            )
                    );
                }


                writer.WriteLine();
            }
        }


        // ==========================================
        // LOAD MATRIX
        // ==========================================

        private double[,] loadMatrix(
            string fileName)
        {
            string[] lines =
                File.ReadAllLines(
                    fileName
                );


            string[] size =
                lines[0]
                    .Split(',');


            int rowCount =
                int.Parse(
                    size[0]
                );


            int columnCount =
                int.Parse(
                    size[1]
                );


            double[,] matrix =
                new double[
                    rowCount,
                    columnCount
                ];


            for (int i = 0;
                 i < rowCount;
                 i++)
            {
                string[] values =
                    lines[i + 1]
                        .Split(',');


                for (int j = 0;
                     j < columnCount;
                     j++)
                {
                    matrix[i, j] =
                        double.Parse(
                            values[j],
                            CultureInfo
                                .InvariantCulture
                        );
                }
            }


            return matrix;
        }


        // ==========================================
        // FORMAT
        // ==========================================

        private static string format(
            double value)
        {
            return value.ToString(
                "0.####",
                CultureInfo
                    .InvariantCulture
            );
        }


        // ==========================================
        // DISPLAY DATAGRID
        // ==========================================

        private void displayResults(
            bool trained)
        {
            if (
                nn3SO == null
                || inputs.Length == 0)
            {
                return;
            }


            rows.Clear();


            int rowCount =
                Math.Max(
                    inodes,
                    Math.Max(
                        hnodes,
                        onodes
                    )
                );


            for (int i = 0;
                 i < rowCount;
                 i++)
            {
                bool isInput =
                    i < inodes;

                bool isHidden =
                    i < hnodes;

                bool isOutput =
                    i < onodes;


                nodeRow data =
                    new nodeRow
                    {
                        inputValue =
                            isInput
                                ? format(
                                    inputs[i]
                                )
                                : "",


                        // Große Matrizen
                        // nicht anzeigen
                        weightsIH =
                            "",


                        inputHidden =
                            isHidden
                                ? format(
                                    nn3SO
                                        .Hidden_inputs[i]
                                )
                                : "",


                        outputHidden =
                            isHidden
                                ? format(
                                    nn3SO
                                        .Hidden_outputs[i]
                                )
                                : "",


                        weightsHO =
                            "",


                        errorHidden =
                            isHidden
                            && trained
                            && nn3SO
                                .Hidden_errors
                                .Length > i

                                ? format(
                                    nn3SO
                                        .Hidden_errors[i]
                                )

                                : "",


                        inputOutput =
                            isOutput

                                ? format(
                                    nn3SO
                                        .Final_inputs[i]
                                )

                                : "",


                        outputLayer =
                            isOutput

                                ? format(
                                    nn3SO
                                        .Final_outputs[i]
                                )

                                : "",


                        target =
                            isOutput
                            && trained
                            && targets.Length > i

                                ? format(
                                    targets[i]
                                )

                                : "",


                        errorOutput =
                            isOutput
                            && trained
                            && nn3SO
                                .Output_errors
                                .Length > i

                                ? format(
                                    nn3SO
                                        .Output_errors[i]
                                )

                                : ""
                    };


                rows.Add(
                    data
                );
            }
        }
    }


    public class nodeRow
    {
        public string inputValue
        {
            get;
            set;
        } = "";


        public string weightsIH
        {
            get;
            set;
        } = "";


        public string inputHidden
        {
            get;
            set;
        } = "";


        public string outputHidden
        {
            get;
            set;
        } = "";


        public string weightsHO
        {
            get;
            set;
        } = "";


        public string errorHidden
        {
            get;
            set;
        } = "";


        public string inputOutput
        {
            get;
            set;
        } = "";


        public string outputLayer
        {
            get;
            set;
        } = "";


        public string target
        {
            get;
            set;
        } = "";


        public string errorOutput
        {
            get;
            set;
        } = "";
    }
}