const fs = require('fs');
const path = require('path');

// 源目录和目标目录
const srcDir = path.resolve(__dirname, '../../dyn_pkg');
const destDir = path.resolve(__dirname, '../dist/demo');

// 递归复制文件夹
function copyFolderSync(src, dest) {
    // 创建目标文件夹
    if (!fs.existsSync(dest)) {
        fs.mkdirSync(dest, { recursive: true });
    }

    // 读取源文件夹内容
    const entries = fs.readdirSync(src, { withFileTypes: true });

    for (const entry of entries) {
        const srcPath = path.join(src, entry.name);
        const destPath = path.join(dest, entry.name);

        if (entry.isDirectory()) {
            // 递归复制子文件夹
            copyFolderSync(srcPath, destPath);
        } else {
            // 复制文件
            fs.copyFileSync(srcPath, destPath);
            console.log(`✅ Copied: ${entry.name}`);
        }
    }
}

console.log('📦 Copying demo data to dist/demo...');
console.log(`Source: ${srcDir}`);
console.log(`Destination: ${destDir}\n`);

try {
    copyFolderSync(srcDir, destDir);
    console.log('\n✨ Demo data copied successfully!');
} catch (error) {
    console.error('❌ Error copying demo data:', error);
    process.exit(1);
}
