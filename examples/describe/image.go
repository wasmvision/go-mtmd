package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
)

func obtainFile(url string) (string, error) {
	if _, err := os.Stat(url); err == nil {
		return copyLocalFileToTemp(url)
	}

	return downloadFile(url)
}

func downloadFile(url string) (string, error) {
	client := &http.Client{}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return "", err
	}

	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8")
	req.Header.Set("Accept-Language", "en-US,en;q=0.5")
	req.Header.Set("Referer", url)

	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("failed to download file: %s", resp.Status)
	}

	fileName := path.Base(url)

	tempFile, err := os.CreateTemp("", fmt.Sprintf("download-*%s", path.Ext(fileName)))
	if err != nil {
		return "", err
	}
	defer tempFile.Close()

	_, err = io.Copy(tempFile, resp.Body)
	if err != nil {
		return "", err
	}

	return tempFile.Name(), nil
}

func copyLocalFileToTemp(filePath string) (string, error) {
	srcFile, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer srcFile.Close()

	tempFile, err := os.CreateTemp("", fmt.Sprintf("download-*%s", path.Ext(filePath)))
	if err != nil {
		return "", err
	}
	defer tempFile.Close()

	_, err = io.Copy(tempFile, srcFile)
	if err != nil {
		return "", err
	}

	return tempFile.Name(), nil
}
