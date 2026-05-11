import fs from 'fs';
import path from 'path';

const MODEL_EXTENSIONS = new Set([
    '.dyn.json',
    '.lcc',
    '.ply',
    '.sog',
    '.sog4d',
    '.splat'
]);

const toPosixPath = value => value.split(path.sep).join('/');

const encodeUrlPath = relativePath => relativePath.split('/').map(encodeURIComponent).join('/');

const isModelEntry = (relativePath) => {
    const lowerPath = relativePath.toLowerCase();
    const baseName = path.posix.basename(lowerPath);

    return baseName === 'meta.json' || Array.from(MODEL_EXTENSIONS).some(ext => lowerPath.endsWith(ext));
};

const scanPublicDirectory = (publicDir) => {
    const files = [];
    const directories = [];
    const models = [];

    if (!fs.existsSync(publicDir)) {
        return { files, directories, models };
    }

    const walk = (directory) => {
        directories.push(directory);

        const children = fs.readdirSync(directory, { withFileTypes: true });
        children.forEach((child) => {
            const absolutePath = path.join(directory, child.name);
            if (child.isDirectory()) {
                walk(absolutePath);
                return;
            }

            if (!child.isFile()) {
                return;
            }

            files.push(absolutePath);

            const relativePath = toPosixPath(path.relative(publicDir, absolutePath));
            if (isModelEntry(relativePath)) {
                models.push({
                    name: path.posix.basename(relativePath),
                    path: relativePath,
                    url: `./${encodeUrlPath(relativePath)}`
                });
            }
        });
    };

    walk(publicDir);

    models.sort((a, b) => a.path.localeCompare(b.path, 'en'));
    files.sort((a, b) => a.localeCompare(b, 'en'));

    return { files, directories, models };
};

export default function publicAssetsAndModelManifest(options = {}) {
    const publicDir = path.resolve(options.publicDir || 'public');
    const manifestFile = options.manifestFile || 'model-manifest.json';

    return {
        name: 'public-assets-and-model-manifest',
        buildStart() {
            const { files, directories } = scanPublicDirectory(publicDir);

            directories.forEach((directory) => {
                this.addWatchFile(directory);
            });

            files.forEach((file) => {
                this.addWatchFile(file);
            });
        },
        generateBundle() {
            const { files, models } = scanPublicDirectory(publicDir);

            files.forEach((file) => {
                const relativePath = toPosixPath(path.relative(publicDir, file));
                this.emitFile({
                    type: 'asset',
                    fileName: relativePath,
                    source: fs.readFileSync(file)
                });
            });

            this.emitFile({
                type: 'asset',
                fileName: manifestFile,
                source: JSON.stringify({ models }, null, 2)
            });
        }
    };
}
