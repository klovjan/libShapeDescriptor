#include <fstream>
#include <cassert>
#include <sstream>
#include <algorithm>
#include <array>
#include <random>
#include <shapeDescriptor/shapeDescriptor.h>
#include <chrono>
#include <iomanip>



std::vector<std::filesystem::path> ShapeDescriptor::listDirectory(const std::filesystem::path& directory) {
    std::vector<std::filesystem::path> foundFiles;

    for(auto &path : std::filesystem::directory_iterator(directory)) {
        foundFiles.emplace_back(path);
    }

    std::sort(foundFiles.begin(), foundFiles.end());

    return foundFiles;
}

std::vector<std::filesystem::path> ShapeDescriptor::listDirectoryAndSubdirectories(const std::filesystem::path &directory) {
    std::vector<std::filesystem::path> foundFiles;

    for(auto &path : std::filesystem::recursive_directory_iterator(directory)) {
        if(path.exists() && path.is_regular_file()) {
            foundFiles.emplace_back(path);
        }
    }

    std::sort(foundFiles.begin(), foundFiles.end());

    return foundFiles;
}

std::vector<char> readLZMAFile(const std::filesystem::path &archiveFile, size_t readLimit, unsigned int threadCount) {
    std::array<char, 5> headerTitle = {0, 0, 0, 0, 0};
    size_t compressedBufferSize;
    size_t decompressedBufferSize;

    if(!std::filesystem::exists(archiveFile)) {
        throw std::runtime_error("The file " + std::filesystem::absolute(archiveFile).string() + " was not found.");
    }

    std::ifstream decompressStream(archiveFile, std::ios::in | std::ios::binary);

    decompressStream.read(headerTitle.data(), 5);
    decompressStream.read((char*) &decompressedBufferSize, sizeof(size_t));
    decompressStream.read((char*) &compressedBufferSize, sizeof(size_t));

    size_t numberOfDecompressedBytesToRead = std::min<size_t>(decompressedBufferSize, readLimit);

    std::vector<char> compressedBuffer(compressedBufferSize);
    std::vector<char> decompressedBuffer(numberOfDecompressedBytesToRead);

    assert(std::string(headerTitle.data()) == "CDXF");

    decompressStream.read(compressedBuffer.data(), compressedBufferSize);

    decompressStream.close();

//#pragma omp critical
    {
        //if(enableMultithreading) {

            //FL2_decompressDCtx(
            //        decompressionContext,
            //        (void*) decompressedBuffer, decompressedBufferSize,
            //        (void*) compressedBuffer, compressedBufferSize);
        //} else {
            ShapeDescriptor::decompressBytesMultithreaded(
                    (void*) decompressedBuffer.data(), numberOfDecompressedBytesToRead,
                    (void*) compressedBuffer.data(), compressedBufferSize,
                    threadCount);
        //}
    }

    return decompressedBuffer;
}

std::vector<char> ShapeDescriptor::readCompressedFile(const std::filesystem::path &archiveFile, unsigned int threadCount) {
    return readLZMAFile(archiveFile, std::numeric_limits<size_t>::max(), threadCount);
}

void ShapeDescriptor::writeCompressedFile(const char *buffer, size_t bufferSize, const std::filesystem::path &archiveFile, unsigned int threadCount) {

    std::filesystem::create_directories(std::filesystem::absolute(archiveFile).parent_path());

    const size_t maxCompressedBufferSize = ShapeDescriptor::computeMaxCompressedBufferSize(bufferSize);
    char* compressedBuffer = new char[maxCompressedBufferSize];
    unsigned long compressedBufferSize;
 //   #pragma omp critical
    {
        compressedBufferSize =
                ShapeDescriptor::compressBytesMultithreaded(
                        (void*) compressedBuffer, maxCompressedBufferSize,
                        (void*) buffer, bufferSize, threadCount);
    }

    const char header[5] = "CDXF";

    std::fstream outStream = std::fstream(archiveFile.string(), std::ios::out | std::ios::binary);

    outStream.write(header, 5 * sizeof(char));
    outStream.write((char*) &bufferSize, sizeof(size_t));
    outStream.write((char*) &compressedBufferSize, sizeof(size_t));
    outStream.write(compressedBuffer, compressedBufferSize);

    outStream.close();

    delete[] compressedBuffer;
}

std::vector<char> ShapeDescriptor::readCompressedFileUpToNBytes(const std::filesystem::path &archiveFile,
                                                         size_t decompressedBytesToRead,
                                                         unsigned int threadCount) {
    return readLZMAFile(archiveFile, decompressedBytesToRead, threadCount);
}

std::string ShapeDescriptor::generateUniqueFilenameString() {
    // Get current time
    auto now = std::chrono::system_clock::now();
    // Get number of milliseconds for the current second
    // (remainder after division into seconds)
    auto ms = duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    // Convert to std::time_t in order to convert to std::tm
    auto timer = std::chrono::system_clock::to_time_t(now);

    // Convert to std::tm (in local time)
    tm bt = *std::localtime(&timer);

    // Create the output time string
    std::stringstream stream;
    stream << std::put_time(&bt, "%Y%m%d-%H%M%S");
    stream << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return stream.str();
}