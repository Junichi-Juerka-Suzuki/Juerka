# CMakeList.txt : Juerka の CMake プロジェクト。ソースを含めて、次を定義します:
# プロジェクト専用ロジックはこちらです。

add_library(Juerka-core STATIC
	"CommonNet/Common.h" "CommonNet/NetworkGroup.h"
	"CommonNet/NetworkGroup.cpp" "CommonNet/SerialNet.h"
	"CommonNet/SerialNet.cpp"
	"Utility/Logger.h" "Utility/Logger.cpp"
	"Utility/TimeRecorder.hpp" "Utility/WeightLogger.cpp"
	"Utility/WeightLogger.h")

target_include_directories(Juerka-core PRIVATE "CommonNet")
target_include_directories(Juerka-core PRIVATE "Utility")

set_property(TARGET Juerka-core PROPERTY CXX_STANDARD 20)

# ソースをこのプロジェクトの実行可能ファイルに追加します。
add_executable(Juerka "test/main.cpp")

target_include_directories(Juerka PRIVATE "CommonNet")
target_include_directories(Juerka PRIVATE "Utility")

set_property(TARGET Juerka PROPERTY CXX_STANDARD 20)

target_link_libraries(Juerka Juerka-core)

# TODO: テストを追加し、必要な場合は、ターゲットをインストールします。
